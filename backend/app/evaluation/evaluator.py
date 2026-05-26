import json
from app.db.sqlite import get_connection
from app.core.interfaces import RetrievalContext
from app.core.retriever import HybridRetriever
from app.core.generator import Generator
from app.ingestion.embedder import OpenAIEmbedder
from app.llm.openai_llm import OpenAILLM
from app.evaluation.metrics import recall_at_k, mrr
from app.evaluation.llm_judge import judge_faithfulness, judge_relevance


class Evaluator:
    def __init__(self):
        self.embedder = OpenAIEmbedder()
        self.llm = OpenAILLM()
        self.retriever = HybridRetriever(embedder=self.embedder)
        self.generator = Generator(llm=self.llm)

    async def run_full_eval(self, kb_id: int) -> dict:
        conn = get_connection()
        kb = conn.execute("SELECT * FROM knowledge_bases WHERE id=?", (kb_id,)).fetchone()
        config = conn.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
        eval_items = conn.execute("SELECT * FROM eval_dataset WHERE kb_id=?", (kb_id,)).fetchall()
        conn.close()

        if not kb or not eval_items:
            return {"error": "No knowledge base or eval data"}

        retrieval_results = []
        generation_results = []

        for item in eval_items:
            relevant_ids = json.loads(item["relevant_doc_ids"] or "[]")
            ctx = RetrievalContext(
                kb_id=kb_id, kb_type=kb["kb_type"],
                access_levels=["public", "internal", "restricted"],
                top_k=config["top_k"] if config else 10,
                retrieval_mode=config["retrieval_mode"] if config else "hybrid",
            )
            chunks = await self.retriever.retrieve(item["question"], ctx)
            retrieved_ids = [c.doc_id for c in chunks]

            retrieval_results.append({
                "question": item["question"],
                "recall@5": recall_at_k(relevant_ids, retrieved_ids, 5),
                "recall@10": recall_at_k(relevant_ids, retrieved_ids, 10),
                "mrr": mrr(relevant_ids, retrieved_ids),
            })

            if item["reference_answer"]:
                result = await self.generator.generate(item["question"], chunks, ctx)
                context_text = "\n".join(c.text for c in chunks[:5])
                faith = await judge_faithfulness(result["answer"], context_text)
                relevance = await judge_relevance(result["answer"], item["question"])
                generation_results.append({
                    "question": item["question"],
                    "faithfulness": round(faith, 3),
                    "relevance": round(relevance, 3),
                })

        avg_recall5 = sum(r["recall@5"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0
        avg_mrr = sum(r["mrr"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0

        return {
            "kb_id": kb_id,
            "retrieval": {
                "avg_recall@5": round(avg_recall5, 4),
                "avg_mrr": round(avg_mrr, 4),
                "details": retrieval_results,
            },
            "generation": generation_results,
        }
