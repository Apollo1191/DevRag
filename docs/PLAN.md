# DevRAG — Project Roadmap (Concise)

## สถานะปัจจุบัน
- โครงโปรเจกต์และ dependencies ถูกสร้างแล้ว (`requirements.txt`).
- Loader: `src/ingestion/loader.py` — clone และเดินไฟล์ (.py, .md) เสร็จแล้ว.
- Heuristic chunker: `src/ingestion/chunker.py` — แบ่งไฟล์ Python/Markdown แบบ heuristics เสร็จแล้ว.

## สิ่งที่ทำแล้ว
1. สร้าง project scaffold และโฟลเดอร์หลัก
2. เพิ่ม `requirements.txt` (เวอร์ชันพินนิ่ง)
3. เขียน `loader.py` (clone repo + walk files)
4. อธิบายเหตุผลการใช้ tree-sitter (การตัด chunk แบบ AST)
5. เขียน heuristic `chunker.py` (Python + Markdown)

## ถัดไป (ลำดับแนะนำ)
6. เขียน unit test สำหรับ `chunker` (ไฟล์เล็กๆ เพื่อยืนยันพฤติกรรม)
7. ผสาน `loader` + `chunker` แบบ end-to-end (clone → walk → chunk)
8. แทนที่ heuristic ด้วย tree-sitter สำหรับ Python/ภาษาอื่นๆ
9. สร้าง `embedder` skeleton (wrapper สำหรับ embeddings พร้อม retry/logging)
10. ติดตั้ง FAISS vector store และโค้ดเก็บ embeddings + metadata
11. สร้าง FastAPI endpoint `POST /query` ที่คืน top-k chunks และ stream คำตอบ
12. เขียน README สรุปวิธีรันและทดสอบ

## Learning checkpoints (เพื่อยืนยันว่าคุณเข้าใจ)
- หลังข้อ 6: อธิบายว่า chunking ทำงานยังไงและแก้กรณี `.ts` ได้อย่างไร
- หลังข้อ 7: อธิบาย flow E2E และยกตัวอย่างผลลัพธ์
- หลังข้อ 8: อธิบายข้อดีของ AST-aware chunking vs fixed-token chunks
- หลังข้อ 10: เปรียบ FAISS vs Qdrant และเหตุผลเลือก

## Quick test (run locally)
- ทดสอบ chunker กับไฟล์ตัวอย่าง:

```powershell
python -m src.ingestion.chunker path\to\some_file.py
```

- ตัวอย่างผลลัพธ์จะแสดง chunk แต่ละชิ้นพร้อมช่วงบรรทัด

## Notes
- คำสั่ง `git` ต้องมีใน PATH เพื่อให้ `loader` ทำงาน
- tree-sitter integration ต้องติดตั้งภาษา/bindings เพิ่ม (จะอธิบายทีละขั้น)

---
Saved: `docs/PLAN.md`
