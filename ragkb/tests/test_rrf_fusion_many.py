from langchain_core.documents import Document

from ragkb.retrievers.rrf_fusion import rrf_fuse_many


def test_rrf_fusion_many():
    a = [Document(page_content="X"), Document(page_content="Y")]
    b = [Document(page_content="Y"), Document(page_content="Z")]

    fused = rrf_fuse_many([a, b], top_k=3)
    assert fused[0].page_content == "Y"  # Y 在两路都出现，分数最高
    assert len(fused) == 3  # X, Y, Z 都在
    assert len({d.page_content for d in fused}) == 3  # 没有重复
