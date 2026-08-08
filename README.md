# AI WhatsApp Message Router
This repository contains a production-grade AI notification router designed to filter the noise of modern messaging platforms. By evaluating multimodal inputs—including transcribing voice notes with OpenAI's Whisper and parsing image context—the system decides whether a message demands immediate attention (notify), can be batched for later (digest), or should be ignored completely (mute).

# Key Features:

 Multimodal Processing: Native handling of texts, images, and audio voice notes.

 Context-Aware Reasoning: Uses relational data (user profiles, group dynamics, business history) to personalize routing decisions.

 Evidence-Based: Retrieves past conversational history to inform current routing actions.

 Safety Guardrails: Hardcoded overrides to instantly mute detected scams, spam, and phishing attempts.
 
 Powered by Groq: Utilizes high-speed Llama 3 models for rapid JSON-structured decision-making.
