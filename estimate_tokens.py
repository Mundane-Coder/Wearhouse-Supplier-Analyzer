from rag_chain import estimate_tokens, index_catalog

index_catalog("sample_catalog.pdf")
print(estimate_tokens("What USB-C cables are available?"))