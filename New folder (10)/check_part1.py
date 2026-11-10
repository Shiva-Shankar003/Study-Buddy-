import db
import vectorstore

print("subjects:", db.list_subjects())
print("chunks stored:", len(db.all_chunks()))

for chunk_id, score in vectorstore.dense_search("what is overfitting", k=3):
    c = db.get_chunks([chunk_id])[0]
    print(round(score, 3), "|", c["ref"], "|", c["text"][:80].replace("\n", ""))