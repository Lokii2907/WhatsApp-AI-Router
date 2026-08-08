# AI WhatsApp Message Router

An AI-powered message routing system built in Python that automatically categorizes multimodal WhatsApp messages (Text, Images, and Voice Notes). 

It uses the Groq API (Llama 3) for reasoning and OpenAI's Whisper for local audio transcription to decide whether a message should be set to `notify`, `digest`, or `mute` based on the user's historical context and safety guardrails.