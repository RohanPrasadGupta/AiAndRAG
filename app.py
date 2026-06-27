import chainlit as cl
from phase3.multi_tool_agent import agent, AgentState

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="""👋 Welcome to the **RAG + Graph Agent**!

I can answer questions using two knowledge sources:
- 📄 **Document search** — definitions, explanations, concepts
- 🕸️ **Graph search** — skills, prerequisites, relationships between concepts

Try asking:
- *"What is Machine Learning?"*
- *"What skills do I need for Deep Learning?"*
- *"What is AI and what skills do I need for ML?"*
"""
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content
    print(f"\n[USER]: {question}")

    async with cl.Step(name="Decomposing question", type="tool") as step:
        step.input = question
        print("[STEP 1] Decomposing question...")

    initial_state: AgentState = {
        "question": question,
        "sub_questions": [],
        "vector_context": "",
        "graph_context": "",
        "combined_context": "",
        "answer": "",
        "tools_tried": []
    }

    print("[STEP 2] Running agent...")
    final_state = await cl.make_async(agent.invoke)(initial_state)
    print(f"[STEP 3] Agent done. Tools used: {final_state.get('tools_tried')}")
    print(f"[STEP 4] Sub-questions: {final_state.get('sub_questions')}")
    print(f"[STEP 5] Answer: {final_state.get('answer')[:100]}...")

    tools = final_state.get("tools_tried", [])
    sub_qs = final_state.get("sub_questions", [])

    async with cl.Step(name=f"Tools used: {', '.join(tools)}", type="tool") as step:
        step.output = "\n".join(
            f"• [{sq['tool']}] {sq['question']}" for sq in sub_qs
        )

    await cl.Message(content=final_state["answer"]).send()
    print("[DONE] Response sent to UI")