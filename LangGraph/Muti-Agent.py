import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import time
from tabulate import tabulate
import json
from typing import Annotated, Literal, Sequence
import operator
from typing_extensions import TypedDict
from langchain.tools import tool

# Set up API keys if not already set
def _set_if_undefined(var: str):
    if var not in os.environ:
        try:
            os.environ[var] = getpass.getpass(f"Enter your {var}: ")
        except KeyboardInterrupt:
            print(f"\nSkipping {var}")
            return False
    return True

# Check required API keys and determine available models
available_models = []

if _set_if_undefined("OPENAI_API_KEY"):
    available_models.extend([
        {"name": "gpt-3.5-turbo", "class": ChatOpenAI, "params": {"model": "gpt-3.5-turbo", "temperature": 0.5}},
        {"name": "gpt-3.5-turbo-16k", "class": ChatOpenAI, "params": {"model": "gpt-3.5-turbo-16k", "temperature": 0.5}},
    ])

if _set_if_undefined("ANTHROPIC_API_KEY"):
    available_models.append(
        {"name": "claude-2", "class": ChatAnthropic, "params": {"model": "claude-2", "temperature": 0.5}}
    )

if not _set_if_undefined("TAVILY_API_KEY"):
    print("Warning: Tavily API key is required for search functionality.")
    exit(1)

# Use available_models instead of the hardcoded models list
models = available_models

if not models:
    print("Error: No models available. Please provide at least one API key.")
    exit(1)

# Set up tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl_tool(code: str):
    """Execute Python code for data visualization. Use matplotlib for plotting and ensure to show the plot."""
    try:
        # Add matplotlib configuration for non-interactive backend
        setup_code = """
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
"""
        if "plt.show()" in code:
            code = code.replace("plt.show()", "plt.savefig('output_plot.png')\nplt.close()")
        
        # Combine setup and user code
        full_code = setup_code + "\n" + code if "import matplotlib" not in code else code
        
        result = repl.run(full_code)
        return f"Code executed successfully. Plot saved as 'output_plot.png'.\nOutput: {result}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Create agent function
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful AI assistant. Use the provided tools to progress towards answering the question. "
         "If you have the final answer, prefix your response with FINAL ANSWER. "
         "You have access to the following tools: {tool_names}.\n{system_message}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
    
    return prompt | llm.bind_tools(tools)

# Define state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def agent_node(state, agent, name):
    try:
        # Get messages from state
        messages = state["messages"]
        
        # Invoke agent
        result = agent.invoke(messages)
        
        # Handle different result types
        if isinstance(result, (list, tuple)):
            result = result[-1]
        
        # Convert to AIMessage if not already a message type
        if not isinstance(result, BaseMessage):
            result = AIMessage(content=str(result))
        
        # Preserve tool calls if present
        if hasattr(result, "tool_calls") and result.tool_calls:
            return {
                "messages": [result],
                "sender": name,
                "tool_calls": result.tool_calls
            }
        
        return {
            "messages": [result],
            "sender": name
        }
    except Exception as e:
        print(f"Error in agent_node: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Error: {str(e)}")],
            "sender": name
        }

def run_test(model_config, test_query):
    start_time = time.time()
    try:
        # Configure research and chart agents
        research_llm = model_config["class"](**model_config["params"])
        chart_llm = model_config["class"](**model_config["params"])

        # Create agents with proper tools
        research_agent = create_agent(research_llm, [tavily_tool], 
            """You are a research assistant. Follow these steps:
            1. Search for accurate data using the tavily_tool
            2. Extract and format the data into a Python-friendly format
            3. When done, pass the formatted data to the Chart_Generator
            4. Do NOT try to create charts yourself""")
        
        chart_agent = create_agent(chart_llm, [python_repl_tool], 
            """You are a data visualization expert. Follow these steps:
            1. Receive the formatted data from the Researcher
            2. Create a clear and informative chart using matplotlib
            3. Include proper labels, title, and save the plot
            4. When the visualization is complete, prefix your response with 'FINAL ANSWER'""")

        # Create agent nodes
        research_node = lambda state: agent_node(state, research_agent, "Researcher")
        chart_node = lambda state: agent_node(state, chart_agent, "Chart_Generator")

        # Set up workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("Researcher", research_node)
        workflow.add_node("Chart_Generator", chart_node)
        
        # Add tool node
        tool_node = ToolNode([tavily_tool, python_repl_tool])
        workflow.add_node("tools", tool_node)

        # Define routing logic
        def router(state: AgentState) -> Literal["tools", "Chart_Generator", "__end__"]:
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            if "FINAL ANSWER" in str(last_message.content):
                return "__end__"
            if state["sender"] == "Researcher":
                return "Chart_Generator"
            return "Researcher"

        # Add edges
        workflow.add_conditional_edges(
            "Researcher",
            router,
            {
                "tools": "tools",
                "Chart_Generator": "Chart_Generator",
                "__end__": END
            }
        )
        
        workflow.add_conditional_edges(
            "Chart_Generator",
            router,
            {
                "tools": "tools",
                "Researcher": "Researcher",
                "__end__": END
            }
        )
        
        workflow.add_conditional_edges(
            "tools",
            lambda x: x["sender"],
            {
                "Researcher": "Researcher",
                "Chart_Generator": "Chart_Generator"
            }
        )
        
        workflow.add_edge(START, "Researcher")
        
        # Compile graph
        graph = workflow.compile()

        # Run workflow with proper initial state
        events = graph.stream(
            {
                "messages": [HumanMessage(content=test_query)],
                "sender": "user"
            },
            {"recursion_limit": 50, "timeout": 60},
            stream_mode="values"
        )

        # Collect results
        success = False
        final_output = ""
        iterations = 0
        for event in events:
            iterations += 1
            if "messages" in event:
                final_output = event["messages"][-1].content
                print(f"Event {iterations}: {final_output[:200]}...")  # Debug output
                if "FINAL ANSWER" in str(final_output):
                    success = True
                    break

        execution_time = time.time() - start_time

        return {
            "success": success,
            "execution_time": execution_time,
            "iterations": iterations,
            "error": "" if success else f"Output incomplete: {final_output[:200]}"
        }

    except Exception as e:
        print(f"Error in run_test: {str(e)}")  # Debug output
        return {
            "success": False,
            "execution_time": time.time() - start_time,
            "iterations": 0,
            "error": str(e)
        }

# Test queries with specific instructions
test_queries = [
    "Search for the US GDP data from 2000 to 2020, then create a line chart using matplotlib with proper labels and title.",
    "Find the market capitalization of the top 5 tech companies in 2023, then create a bar chart using matplotlib to visualize the data.",
    "Search for global CO2 emissions data for the last decade, then create a line plot using matplotlib with trend analysis."
]

# Run tests and collect results
results = []
for model in models:
    model_results = {
        "Model": model["name"],
        "Success Rate": 0,
        "Avg Time": 0,
        "Avg Iterations": 0,
        "Errors": 0
    }

    for query in test_queries:
        result = run_test(model, query)
        if result["success"]:
            model_results["Success Rate"] += 1
        model_results["Avg Time"] += result["execution_time"]
        model_results["Avg Iterations"] += result["iterations"]
        if result["error"]:
            model_results["Errors"] += 1

    # 计算平均值
    total_tests = len(test_queries)
    model_results["Success Rate"] = f"{(model_results['Success Rate']/total_tests)*100:.1f}%"
    model_results["Avg Time"] = f"{model_results['Avg Time']/total_tests:.1f}s"
    model_results["Avg Iterations"] = f"{model_results['Avg Iterations']/total_tests:.1f}"

    results.append(model_results)

# 生成表格
headers = ["Model", "Success Rate", "Avg Time", "Avg Iterations", "Errors"]
table = tabulate(
    [[r["Model"], r["Success Rate"], r["Avg Time"], r["Avg Iterations"], r["Errors"]] for r in results],
    headers=headers,
    tablefmt="grid"
)

print("\nModel Performance Comparison:")
print(table)

# 分析最佳模型
best_model = max(results, key=lambda x: float(x["Success Rate"].rstrip('%')))
print(f"\nBest Performing Model: {best_model['Model']}")
print(f"Success Rate: {best_model['Success Rate']}")
print(f"Average Execution Time: {best_model['Avg Time']}")
print(f"Average Iterations: {best_model['Avg Iterations']}")
