from jud_rev import PDFJuridicoAnalyzer


# Com configurações específicas
analyzer = PDFJuridicoAnalyzer(tesseract_path="C:\Program Files\Tesseract-OCR")

# Análise completa
resultado = analyzer.analisar_pdf_completo("documento.pdf")

# Acessar diferentes resumos
resumo_executivo = resultado['resumos']['executivo']      # Curto (~100 palavras)
resumo_completo = resultado['resumos']['completo']        # Médio (~200 palavras)
resumo_detalhado = resultado['resumos']['detalhado']      # Longo (~300 palavras)

# Entidades jurídicas extraídas
entidades = resultado['entidades_juridicas']
print("Processos:", entidades['numeros_processo'])
print("Leis citadas:", entidades['leis_citadas'])
print("Tribunais:", entidades['tribunais'])

# Estatísticas
stats = resultado['estatisticas']
print(f"Palavras: {stats['total_palavras']}")
print(f"Densidade jurídica: {stats['densidade_juridica']:.2f}%")