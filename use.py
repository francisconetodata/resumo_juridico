from jud_revisor import PDFJuridicoAnalyzer

# Inicializar
analyzer = PDFJuridicoAnalyzer()
tamanho_minimo_desejado = 1000
tamanho_maximo_desejado = 1200
# Analisar PDF
resultado = analyzer.analisar_pdf_completo("documento.pdf",
                                           min_resumo=tamanho_minimo_desejado,
                                           max_resumo=tamanho_maximo_desejado)

# Ver resultados
print(f"Resumo: {resultado['resumos']['executivo']}")