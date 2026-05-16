import whisper
import json

model = whisper.load_model("base")
result = model.transcribe("../france24.mp3")

with open("../transcription.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(result["text"])