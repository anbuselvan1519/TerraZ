# llama_wrapper.py
import os
try:
    import streamlit as st
except ImportError:
    st = None

def get_llama():
    """Initializes or returns cached LLaMA instance."""
    if st is not None and "llama" in st.session_state:
        return st.session_state["llama"]
    try:
        from llama_cpp import Llama
        model_path = r"C:\Users\admin\OneDrive\Documents\projects\llama.cpp\models\llama-3.2-1b-instruct-q4_k_m.gguf"
        if not os.path.exists(model_path):
            return None
        llama_inst = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=max(1, (os.cpu_count() or 4) // 2),
            n_gpu_layers=0,   # CPU safe
            verbose=False
        )
        if st is not None:
            st.session_state["llama"] = llama_inst
        return llama_inst
    except Exception as e:
        if st is not None:
            st.session_state["llama_error"] = str(e)
        return None


def ask_llama(prompt: str, history: list = None) -> str:
    """
    Ask LLaMA a question with conversation history.
    history = list of {"role": ..., "content": ...}
    Includes graceful domain fallback if model fails.
    """
    llm = get_llama()
    
    if llm is None:
        # Fallback intelligent domain response
        p_lower = prompt.lower()
        if "lithium" in p_lower:
            return "Lithium is primarily mined from spodumene pegmatites (e.g. Greenbushes, Pilgangoora in Australia) and continental brine salars (e.g. Salar de Atacama in Chile, Hombre Muerto in Argentina). It is the critical cathode and electrolyte component in LFP and NMC electric vehicle batteries."
        elif "cobalt" in p_lower:
            return "Over 70% of the world's cobalt is extracted from stratiform sediment-hosted copper-cobalt deposits in the Katanga Copperbelt of DR Congo. Cobalt enhances thermal stability and energy density in high-performance NMC and NCA EV battery cathodes."
        elif "nickel" in p_lower:
            return "Nickel is essential for high-energy density EV batteries (NMC811, NCA). Major global reserves are found in ultramafic laterites across Indonesia, the Philippines, and New Caledonia, and in magmatic sulfide deposits in Russia (Norilsk) and Canada (Sudbury)."
        elif "graphite" in p_lower:
            return "Natural flake graphite is the dominant anode material in EV lithium-ion batteries. Leading global suppliers include China, Mozambique (Cabo Delgado / Balama), Madagascar, and Brazil."
        elif "rare earth" in p_lower or "ree" in p_lower:
            return "Rare Earth Elements such as Neodymium (Nd), Praseodymium (Pr), and Dysprosium (Dy) are vital for permanent magnet synchronous motors (PMSM) used in EV drivetrains. Major deposits include Bayan Obo (China), Mountain Pass (USA), and Mount Weld (Australia)."
        elif "copper" in p_lower:
            return "Electric vehicles require approximately four times more copper than conventional internal combustion vehicles (~83 kg vs ~23 kg). Top global copper producers include Chile, Peru, DR Congo, and China."
        else:
            return f"As an EV Mineral Exploration assistant, I specialize in critical transition minerals including Lithium, Cobalt, Nickel, Graphite, Rare Earths, Copper, and Manganese across 50 global jurisdictions. How can I help you explore deposits, grades, or locations?"

    # Ensure no duplicate user prompt at the end
    messages = list(history) if history else []
    if not messages or messages[-1].get("content") != prompt:
        messages.append({"role": "user", "content": prompt})

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating LLaMA response: {e}"
