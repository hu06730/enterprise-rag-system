import logging

import re


logger = logging.getLogger(__name__)


class QueryRewriter:
    """把一个复合问题拆成 2~3 个子查询，供多路召回。"""

    def __init__(self, generator):   # 复用你的 LLMGenerator
        self.generator = generator
        # 行首编号或项目符号：1. / 1、/ 1) / - / *
        self._prefix = re.compile(r"^\s*(?:\d+[\.、\)]|[-*•])\s*")



    def _parse(self,raw : str,query : str,n : int)->list[str]:
        lines = []
        for line in raw.splitlines():
            s = self._prefix.sub("", line).strip()
            s = s.strip("“”\"'「」")
            if len(s)<3:
                continue
            lines.append(s)

        # 去重但保持顺序（set 会打乱顺序，别直接用 set）
        seen,uniq  =set(),[]
        for s in lines:
            if s not in seen:
                seen.add(s)
                uniq.append(s)

        # 原问题必须排第一：模型没输出它时补上
        if not uniq or uniq[0] != query:
            uniq = [query] + [s for s in uniq if s != query]

        return uniq[:max(1,n)]



    def rewrite(self, query: str, n: int = 3) -> list[str]:
        """返回 [原 query, 子查询1, 子查询2] —— 原 query 必须保留。

        失败/超时/返回格式不对 → 返回 [query]（兜底，不能让检索崩）
        """
        if not query.strip():
            return []

        REWRITE_PROMPT = f'''你是检索查询改写助手。
把用户问题改写成最多{n}个更短、更具体的子问题，要求每个子问题都能脱离上下文单独检索。

硬性要求：
1.第一行必须原样输出用户问题。
2.每行只输出一个子问题，不要编号、不要引号、不要解释、不要空行。
3.保留原问题中的专有名词（例如timeit、doctest、pip等），不要换成同义词

用户问题：{query}
'''
        try:
            results = self.generator.generate(REWRITE_PROMPT)
            return self._parse(results,query,n) or [query]
        except Exception as e:
            logger.warning("查询改写失败，回退原问题: %s", e)
            return [query]




if __name__ == "__main__":
    # 调试用的假 generator 定义在这里（自包含）：
    # 生产代码绝不能 import tests/ —— 否则删测试数据会让生产代码崩
    class _LocalFakeGen:
        def generate(self, prompt: str) -> str:
            print("=== 实际发出的 PROMPT ===")
            print(prompt)
            print("=== END ===")
            return "1. 怎么测量代码运行时间\n2. 「doctest 怎么写测试」\n\n以上是两个子问题"

    r = QueryRewriter(_LocalFakeGen())
    print(r.rewrite("想知道两段代码哪个跑得更快，还想在文档注释里顺手写点测试，分别该用哪个标准库"))
