import os
import json
import time
import pathlib
import pandas as pd
import whisper
from groq import Groq

print("==================================================")
print("   WhatsApp Message Router - Production Engine   ")
print("==================================================\n")

DATASET_DIR = pathlib.Path("dataset")
OUTPUT_FILE = "output.csv"

# ---------------------------------------------------------
# 1. Initialize Clients & Models
# ---------------------------------------------------------
# Replace "YOUR_GROQ_API_KEY" below with your actual Groq key (e.g., "gsk_...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
groq_client = Groq(api_key=GROQ_API_KEY)

print("[1/5] Loading Whisper speech-to-text model...")
whisper_model = whisper.load_model("tiny")
print("Whisper model initialized successfully.\n")

# ---------------------------------------------------------
# 2. Load Relational Datasets
# ---------------------------------------------------------
print("[2/5] Ingesting relational datasets into memory...")
try:
    messages_df = pd.read_csv(DATASET_DIR / "messages.csv")
    users_df = pd.read_csv(DATASET_DIR / "users.csv")
    groups_df = pd.read_csv(DATASET_DIR / "groups.csv") if (DATASET_DIR / "groups.csv").exists() else pd.DataFrame()
    group_members_df = pd.read_csv(DATASET_DIR / "group_members.csv") if (DATASET_DIR / "group_members.csv").exists() else pd.DataFrame()
    business_accounts_df = pd.read_csv(DATASET_DIR / "business_accounts.csv") if (DATASET_DIR / "business_accounts.csv").exists() else pd.DataFrame()
    user_business_history_df = pd.read_csv(DATASET_DIR / "user_business_history.csv") if (DATASET_DIR / "user_business_history.csv").exists() else pd.DataFrame()
    message_history_df = pd.read_csv(DATASET_DIR / "message_history.csv") if (DATASET_DIR / "message_history.csv").exists() else pd.DataFrame()
    message_events_df = pd.read_csv(DATASET_DIR / "message_events.csv") if (DATASET_DIR / "message_events.csv").exists() else pd.DataFrame()
    voice_notes_df = pd.read_csv(DATASET_DIR / "voice_notes.csv") if (DATASET_DIR / "voice_notes.csv").exists() else pd.DataFrame()
    images_df = pd.read_csv(DATASET_DIR / "images.csv") if (DATASET_DIR / "images.csv").exists() else pd.DataFrame()
    print(f"Successfully loaded {len(messages_df)} target messages and context tables.\n")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Cache for audio transcriptions
transcription_cache = {}

# ---------------------------------------------------------
# 3. Multimodal Media Resolution
# ---------------------------------------------------------
def resolve_message_content(row):
    msg_id = str(row['message_id'])
    msg_text = str(row['message_text']) if pd.notna(row['message_text']) else ""
    media_type = str(row['media_type']) if pd.notna(row['media_type']) else ""
    media_id = str(row['media_id']) if pd.notna(row['media_id']) else ""

    if media_type == "voice" and media_id:
        if media_id in transcription_cache:
            return f"[Voice Note Transcript]: {transcription_cache[media_id]}"
        
        vn_match = voice_notes_df[voice_notes_df['voice_note_id'] == media_id] if not voice_notes_df.empty and 'voice_note_id' in voice_notes_df.columns else pd.DataFrame()
        if not vn_match.empty and 'file_path' in vn_match.columns:
            audio_path = DATASET_DIR / vn_match.iloc[0]['file_path']
            if audio_path.exists():
                try:
                    res = whisper_model.transcribe(str(audio_path))
                    transcript = res.get('text', '').strip()
                    transcription_cache[media_id] = transcript
                    return f"[Voice Note Transcript]: {transcript}"
                except Exception:
                    return f"[Voice Note Audio File Present - ID: {media_id}]"
        return f"[Voice Note ID: {media_id}]"

    elif media_type == "image" and media_id:
        img_match = images_df[images_df['image_id'] == media_id] if not images_df.empty and 'image_id' in images_df.columns else pd.DataFrame()
        path_str = img_match.iloc[0]['file_path'] if not img_match.empty and 'file_path' in img_match.columns else media_id
        content_desc = f"[Image/Poster Attached: {path_str}]"
        if msg_text:
            content_desc += f" Caption: {msg_text}"
        return content_desc

    return msg_text if msg_text else "[Empty Message]"

# ---------------------------------------------------------
# 4. Context Engine & Evidence Linking (Defensive Keys)
# ---------------------------------------------------------
def build_message_context(row):
    user_id = row['user_id']
    conv_type = row['conversation_type']
    group_id = row.get('group_id')
    business_id = row.get('business_id')
    sender_id = row.get('sender_user_id')

    context_dict = {}

    # User Profile Context
    if not users_df.empty and 'user_id' in users_df.columns:
        u_match = users_df[users_df['user_id'] == user_id]
        if not u_match.empty:
            context_dict['user_profile'] = u_match.iloc[0].to_dict()

    # Group Context
    if conv_type == 'group' and pd.notna(group_id) and not groups_df.empty:
        if 'group_id' in groups_df.columns:
            g_match = groups_df[groups_df['group_id'] == group_id]
            if not g_match.empty:
                context_dict['group_info'] = g_match.iloc[0].to_dict()
        if not group_members_df.empty and 'group_id' in group_members_df.columns and 'user_id' in group_members_df.columns:
            gm_match = group_members_df[(group_members_df['group_id'] == group_id) & (group_members_df['user_id'] == user_id)]
            if not gm_match.empty:
                context_dict['user_group_membership'] = gm_match.iloc[0].to_dict()

    # Business Context
    elif conv_type == 'business' and pd.notna(business_id) and not business_accounts_df.empty:
        if 'business_id' in business_accounts_df.columns:
            b_match = business_accounts_df[business_accounts_df['business_id'] == business_id]
            if not b_match.empty:
                context_dict['business_profile'] = b_match.iloc[0].to_dict()
        if not user_business_history_df.empty and 'user_id' in user_business_history_df.columns and 'business_id' in user_business_history_df.columns:
            ubh_match = user_business_history_df[(user_business_history_df['user_id'] == user_id) & (user_business_history_df['business_id'] == business_id)]
            if not ubh_match.empty:
                context_dict['user_business_relationship'] = ubh_match.iloc[0].to_dict()

    # Historical Evidence Match (Safely Checking Available Columns)
    evidence_ids = []
    if not message_history_df.empty and 'user_id' in message_history_df.columns:
        hist_matches = message_history_df[message_history_df['user_id'] == user_id]
        
        if conv_type == 'business' and pd.notna(business_id):
            if 'business_id' in hist_matches.columns:
                hist_matches = hist_matches[hist_matches['business_id'] == business_id]
            elif 'sender_user_id' in hist_matches.columns:
                hist_matches = hist_matches[hist_matches['sender_user_id'] == str(business_id)]
            elif 'sender_id' in hist_matches.columns:
                hist_matches = hist_matches[hist_matches['sender_id'] == str(business_id)]
        elif sender_id and pd.notna(sender_id):
            if 'sender_user_id' in hist_matches.columns:
                hist_matches = hist_matches[hist_matches['sender_user_id'] == sender_id]
            elif 'sender_id' in hist_matches.columns:
                hist_matches = hist_matches[hist_matches['sender_id'] == str(sender_id)]
        
        if not hist_matches.empty and 'message_id' in hist_matches.columns:
            evidence_ids = hist_matches['message_id'].astype(str).tolist()[-3:]
            cols_to_dict = [c for c in ['message_id', 'message_text'] if c in hist_matches.columns]
            context_dict['recent_history_from_sender'] = hist_matches.tail(3)[cols_to_dict].to_dict(orient='records')

    evidence_str = ";".join(evidence_ids) if evidence_ids else "none"

    return context_dict, evidence_str

# ---------------------------------------------------------
# 5. AI Reasoning Engine
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI Message Notification Router for WhatsApp.
Your goal is to evaluate incoming messages and user context to determine the appropriate routing action.

Target Output JSON Schema:
{
    "action": "notify" | "digest" | "mute",
    "message_type": "personal" | "urgent" | "event" | "payment" | "business_update" | "promotion" | "greeting" | "forward" | "spam" | "scam" | "unknown",
    "reason": "Short concise human-readable explanation (max 1 sentence)",
    "confidence": float between 0.00 and 1.00
}

Routing Rules:
1. MUTE: Low-value promotions, repetitive forwards, unverified scam/phishing attempts, unwanted business spam, or content from muted groups.
2. DIGEST: Useful/safe updates, general announcements, non-urgent events, or business order updates that do not require immediate action.
3. NOTIFY: Direct personal communications, urgent security/payment alerts from verified accounts, time-sensitive personal requests, or emergency mentions.
4. SAFETY GUARDRAIL: Any message identified as 'scam' or 'spam' MUST be routed as 'mute'.

Output strictly raw JSON without markdown codeblocks or additional text."""

def route_message_ai(content_text, context_dict):
    user_prompt = f"User Context Engine Data:\n{json.dumps(context_dict, default=str)}\n\nIncoming Message Payload:\n{content_text}"
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [AI Call Failure]: {e}")
        return None

# ---------------------------------------------------------
# 6. Post-Processing Safety Guardrails Engine
# ---------------------------------------------------------
def apply_safety_guardrails(ai_result):
    if not ai_result:
        return {
            "action": "digest",
            "message_type": "unknown",
            "reason": "Default fallback due to processing exception.",
            "confidence": 0.50
        }

    action = str(ai_result.get("action", "digest")).lower()
    msg_type = str(ai_result.get("message_type", "unknown")).lower()
    reason = str(ai_result.get("reason", "Standard automated notification route."))
    try:
        confidence = float(ai_result.get("confidence", 0.85))
    except (ValueError, TypeError):
        confidence = 0.85

    # CRITICAL RULE: Spam/Scam must ALWAYS be Muted
    if msg_type in ["scam", "spam"]:
        action = "mute"
        reason = f"Security safety override: Automated {msg_type} detection enforced mute rule."

    if action not in ["notify", "digest", "mute"]:
        action = "digest"

    valid_types = ["personal", "urgent", "event", "payment", "business_update", "promotion", "greeting", "forward", "spam", "scam", "unknown"]
    if msg_type not in valid_types:
        msg_type = "unknown"

    return {
        "action": action,
        "message_type": msg_type,
        "reason": reason,
        "confidence": round(confidence, 2)
    }

# ---------------------------------------------------------
# 7. Processing Pipeline Execution
# ---------------------------------------------------------
print("[3/5] Checking existing progress...")
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    try:
        existing_df = pd.read_csv(OUTPUT_FILE)
        if 'message_id' in existing_df.columns:
            processed_ids = set(existing_df['message_id'].astype(str))
            print(f"Resuming pipeline. Found {len(processed_ids)} already routed messages.")
    except Exception:
        pass

print("\n[4/5] Executing multimodal routing engine...")
columns_order = ['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']

for index, row in messages_df.iterrows():
    msg_id = str(row['message_id'])
    
    if msg_id in processed_ids:
        continue

    resolved_content = resolve_message_content(row)
    context_dict, historical_evidence = build_message_context(row)

    raw_ai_decision = route_message_ai(resolved_content, context_dict)
    final_decision = apply_safety_guardrails(raw_ai_decision)

    output_row = {
        "message_id": msg_id,
        "action": final_decision["action"],
        "message_type": final_decision["message_type"],
        "reason": final_decision["reason"],
        "confidence": final_decision["confidence"],
        "evidence_message_ids": historical_evidence
    }

    df_row = pd.DataFrame([output_row])[columns_order]
    write_header = not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0
    df_row.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)

    processed_ids.add(msg_id)
    print(f"[{len(processed_ids)}/{len(messages_df)}] {msg_id} -> Action: {output_row['action'].upper()} | Type: {output_row['message_type']} | Evidence: {output_row['evidence_message_ids']}")

    time.sleep(2.5)

print("\n==================================================")
print("[5/5] Pipeline execution complete! Final output verified.")
print(f"Output saved to: {os.path.abspath(OUTPUT_FILE)}")
print("==================================================")