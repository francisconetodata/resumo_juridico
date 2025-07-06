#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador de PDFs Jurídicos com IA e OCR
Análise completa de documentos jurídicos usando modelos de IA gratuitos
"""

import os
import re
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel, AutoModelForCausalLM
)
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFJuridicoAnalyzer:
    """
    Classe principal para análise de PDFs jurídicos com IA e OCR
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Inicializa o analisador
        
        Args:
            tesseract_path: Caminho para o executável do Tesseract (opcional)
        """
        # Configurar Tesseract se caminho fornecido
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Configurar dispositivo (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar modelos
        self._inicializar_modelos()
        
        # Configurações para documentos jurídicos
        self.termos_juridicos = [
            'sentença', 'acórdão', 'despacho', 'decisão', 'recurso',
            'apelação', 'embargos', 'agravo', 'mandado', 'habeas corpus',
            'ação', 'processo', 'autor', 'réu', 'juiz', 'desembargador',
            'tribunal', 'vara', 'comarca', 'foro', 'instância',
            'código civil', 'código penal', 'constituição', 'lei',
            'decreto', 'medida provisória', 'jurisprudência'
        ]
        
    def _inicializar_modelos(self):
        """Inicializa os modelos de IA necessários"""
        try:
            logger.info("Carregando modelos de IA...")
            
            # Modelo para classificação e análise em português
            self.tokenizer_bert = AutoTokenizer.from_pretrained(
                "neuralmind/bert-base-portuguese-cased"
            )
            self.model_bert = AutoModel.from_pretrained(
                "neuralmind/bert-base-portuguese-cased"
            ).to(self.device)
            
            # Pipeline para resumos
            self.summarizer = pipeline(
                "summarization",
                model="pierreguillou/gpt2-small-portuguese",
                tokenizer="pierreguillou/gpt2-small-portuguese",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Pipeline para análise de sentimentos
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Modelos carregados com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
            # Fallback para modelos básicos
            self._carregar_modelos_fallback()
    
    def _carregar_modelos_fallback(self):
        """Carrega modelos de fallback em caso de erro"""
        logger.info("Carregando modelos de fallback...")
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelos de fallback: {e}")
    
    def extrair_texto_pdf(self, caminho_pdf: str) -> str:
        """
        Extrai texto de um arquivo PDF
        
        Args:
            caminho_pdf: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído do PDF
        """
        texto_completo = ""
        
        try:
            documento = fitz.open(caminho_pdf)
            logger.info(f"PDF aberto: {len(documento)} páginas")
            
            for num_pagina in range(len(documento)):
                pagina = documento[num_pagina]
                
                # Tentar extrair texto direto
                texto_pagina = pagina.get_text()
                
                # Se não há texto, fazer OCR
                if not texto_pagina.strip():
                    logger.info(f"Fazendo OCR na página {num_pagina + 1}")
                    texto_pagina = self._ocr_pagina(pagina)
                
                texto_completo += f"\n--- PÁGINA {num_pagina + 1} ---\n{texto_pagina}\n"
            
            documento.close()
            
        except Exception as e:
            logger.error(f"Erro ao extrair texto do PDF: {e}")
            return ""
        
        return texto_completo
    
    def _ocr_pagina(self, pagina) -> str:
        """
        Realiza OCR em uma página do PDF
        
        Args:
            pagina: Objeto da página do PyMuPDF
            
        Returns:
            Texto extraído via OCR
        """
        try:
            # Converter página para imagem
            matriz = fitz.Matrix(2.0, 2.0)  # Aumentar resolução
            pixmap = pagina.get_pixmap(matrix=matriz)
            img_data = pixmap.tobytes("png")
            
            # Carregar imagem com PIL
            imagem = Image.open(io.BytesIO(img_data))
            
            # Configurar OCR para português
            config_ocr = '--oem 3 --psm 6 -l por'
            
            # Realizar OCR
            texto = pytesseract.image_to_string(imagem, config=config_ocr)
            
            return texto
            
        except Exception as e:
            logger.error(f"Erro no OCR: {e}")
            return ""
    
    def preprocessar_texto(self, texto: str) -> str:
        """
        Preprocessa o texto para análise
        
        Args:
            texto: Texto bruto extraído
            
        Returns:
            Texto preprocessado
        """
        # Remover quebras de linha excessivas
        texto = re.sub(r'\n+', '\n', texto)
        
        # Remover espaços excessivos
        texto = re.sub(r' +', ' ', texto)
        
        # Remover caracteres especiais desnecessários
        texto = re.sub(r'[^\w\s\.,;:!?()-]', '', texto)
        
        # Capitalizar início de sentenças
        texto = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), texto)
        
        return texto.strip()
    
    def dividir_texto_chunks(self, texto: str, tamanho_max: int = 1000) -> List[str]:
        """
        Divide texto em chunks para processamento
        
        Args:
            texto: Texto completo
            tamanho_max: Tamanho máximo de cada chunk
            
        Returns:
            Lista de chunks de texto
        """
        palavras = texto.split()
        chunks = []
        chunk_atual = []
        tamanho_atual = 0
        
        for palavra in palavras:
            if tamanho_atual + len(palavra) + 1 <= tamanho_max:
                chunk_atual.append(palavra)
                tamanho_atual += len(palavra) + 1
            else:
                if chunk_atual:
                    chunks.append(' '.join(chunk_atual))
                chunk_atual = [palavra]
                tamanho_atual = len(palavra)
        
        if chunk_atual:
            chunks.append(' '.join(chunk_atual))
        
        return chunks
    
    def extrair_entidades_juridicas(self, texto: str) -> Dict[str, List[str]]:
        """
        Extrai entidades jurídicas específicas do texto
        
        Args:
            texto: Texto para análise
            
        Returns:
            Dicionário com entidades encontradas
        """
        entidades = {
            'numeros_processo': [],
            'leis_citadas': [],
            'jurisprudencia': [],
            'pessoas_envolvidas': [],
            'tribunais': [],
            'termos_juridicos': []
        }
        
        # Números de processo
        padrao_processo = r'\b\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\.\d{6}-\d{1}\b'
        entidades['numeros_processo'] = re.findall(padrao_processo, texto)
        
        # Leis citadas
        padrao_lei = r'Lei\s+n[ºo°]?\s*\d+[\./]\d+|Art\.\s*\d+|Artigo\s+\d+'
        entidades['leis_citadas'] = re.findall(padrao_lei, texto, re.IGNORECASE)
        
        # Tribunais
        padrao_tribunal = r'TJ[A-Z]{2}|STF|STJ|TST|TSE|TRF|TRT|Tribunal|Vara|Comarca'
        entidades['tribunais'] = re.findall(padrao_tribunal, texto, re.IGNORECASE)
        
        # Termos jurídicos específicos
        for termo in self.termos_juridicos:
            if termo.lower() in texto.lower():
                entidades['termos_juridicos'].append(termo)
        
        return entidades
    
    def gerar_resumo(self, texto: str, tipo_resumo: str = "completo", 
                     min_length: Optional[int] = None, max_length: Optional[int] = None) -> str:
        """
        Gera resumo do texto usando IA
        
        Args:
            texto: Texto para resumir
            tipo_resumo: Tipo de resumo ("completo", "executivo", "detalhado")
            
        Returns:
            Resumo gerado
        """
        try:
            chunks = self.dividir_texto_chunks(texto, 900)
            resumos_parciais = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Resumindo chunk {i+1}/{len(chunks)} para resumo '{tipo_resumo}'")
                
                # --- ALTERAÇÃO AQUI ---
                # Define os tamanhos do resumo. Usa os valores customizados se forem fornecidos,
                # caso contrário, usa os padrões baseados no tipo_resumo.
                if min_length is not None and max_length is not None:
                    final_max_length = max_length
                    final_min_length = min_length
                else:
                    if tipo_resumo == "executivo":
                        final_max_length = 100
                        final_min_length = 50
                    elif tipo_resumo == "detalhado":
                        final_max_length = 300
                        final_min_length = 150
                    else:  # completo
                        final_max_length = 200
                        final_min_length = 100
                # --- FIM DA ALTERAÇÃO ---

                try:
                    resumo = self.summarizer(
                        chunk,
                        max_length=final_max_length,
                        min_length=final_min_length,
                        do_sample=False
                    )
                    resumos_parciais.append(resumo[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Erro ao resumir chunk {i+1}: {e}")
                    sentencas = chunk.split('.')[:3]
                    resumos_parciais.append('. '.join(sentencas) + '.')
            
            resumo_final = ' '.join(resumos_parciais)
            
            # Se o resumo combinado for muito longo, resume novamente
            if len(resumo_final) > 1000 and len(resumos_parciais) > 1:
                try:
                    resumo_final = self.summarizer(
                        resumo_final,
                        max_length=500,
                        min_length=200,
                        do_sample=False
                    )[0]['summary_text']
                except:
                    pass
            
            return resumo_final
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {e}")
            # Fallback: resumo manual
            return self._resumo_manual(texto)
    
    def _resumo_manual(self, texto: str) -> str:
        """
        Gera resumo manual em caso de falha da IA
        
        Args:
            texto: Texto para resumir
            
        Returns:
            Resumo manual
        """
        sentencas = texto.split('.')
        resumo = []
        
        # Pegar primeiras sentenças de cada parágrafo
        paragrafos = texto.split('\n')
        for paragrafo in paragrafos[:5]:
            primeira_sentenca = paragrafo.split('.')[0]
            if len(primeira_sentenca) > 20:
                resumo.append(primeira_sentenca + '.')
        
        return ' '.join(resumo)
    
    def analisar_sentimento(self, texto: str) -> Dict[str, any]:
        """
        Analisa o sentimento/tom do documento
        
        Args:
            texto: Texto para análise
            
        Returns:
            Resultado da análise de sentimento
        """
        try:
            chunks = self.dividir_texto_chunks(texto, 500)
            sentimentos = []
            
            for chunk in chunks[:5]:  # Analisar apenas primeiros 5 chunks
                resultado = self.sentiment_analyzer(chunk)
                sentimentos.append(resultado[0])
            
            # Agregar resultados
            scores_positivos = [s['score'] for s in sentimentos if s['label'] in ['POSITIVE', 'POS']]
            scores_negativos = [s['score'] for s in sentimentos if s['label'] in ['NEGATIVE', 'NEG']]
            scores_neutros = [s['score'] for s in sentimentos if s['label'] in ['NEUTRAL', 'NEU']]
            
            return {
                'tom_geral': 'positivo' if len(scores_positivos) > len(scores_negativos) else 'negativo' if len(scores_negativos) > len(scores_positivos) else 'neutro',
                'confianca_positiva': np.mean(scores_positivos) if scores_positivos else 0,
                'confianca_negativa': np.mean(scores_negativos) if scores_negativos else 0,
                'confianca_neutra': np.mean(scores_neutros) if scores_neutros else 0
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimento: {e}")
            return {'tom_geral': 'indeterminado', 'erro': str(e)}
    
    def classificar_documento(self, texto: str, entidades: Dict) -> str:
        """
        Classifica o tipo de documento jurídico
        
        Args:
            texto: Texto do documento
            entidades: Entidades extraídas
            
        Returns:
            Classificação do documento
        """
        texto_lower = texto.lower()
        
        # Padrões para diferentes tipos de documentos
        padroes = {
            'Sentença': ['sentença', 'julgo', 'dispositivo', 'condeno'],
            'Acórdão': ['acórdão', 'acordam', 'turma', 'relator'],
            'Despacho': ['despacho', 'intime-se', 'cumpra-se'],
            'Petição': ['petição', 'requer', 'solicita', 'postula'],
            'Contrato': ['contrato', 'partes', 'cláusula', 'objeto'],
            'Recurso': ['recurso', 'apelação', 'agravo', 'embargos'],
            'Mandado': ['mandado', 'oficial de justiça', 'cumprir'],
        }
        
        pontuacoes = {}
        for tipo, palavras in padroes.items():
            pontuacao = sum(texto_lower.count(palavra) for palavra in palavras)
            pontuacoes[tipo] = pontuacao
        
        if max(pontuacoes.values()) > 0:
            return max(pontuacoes, key=pontuacoes.get)
        else:
            return 'Documento Jurídico Genérico'
    
    def analisar_pdf_completo(self, caminho_pdf: str, 
                              min_resumo: Optional[int] = None, 
                              max_resumo: Optional[int] = None) -> Dict[str, any]:
        """
        Realiza análise completa do PDF
        
        Args:
            caminho_pdf: Caminho para o arquivo PDF
            min_resumo: Tamanho mínimo para um resumo personalizado (opcional)
            max_resumo: Tamanho máximo para um resumo personalizado (opcional)
            
        Returns:
            Dicionário com análise completa
        """
        logger.info(f"Iniciando análise completa do PDF: {caminho_pdf}")
        
        # 1. Extrair texto
        texto_bruto = self.extrair_texto_pdf(caminho_pdf)
        if not texto_bruto:
            return {'erro': 'Não foi possível extrair texto do PDF'}
        
        # 2. Preprocessar texto
        texto_limpo = self.preprocessar_texto(texto_bruto)
        
        # 3. Extrair entidades jurídicas
        entidades = self.extrair_entidades_juridicas(texto_limpo)
        
        # 4. Classificar documento
        tipo_documento = self.classificar_documento(texto_limpo, entidades)
        
        # 5. Gerar resumos
        resumo_executivo = self.gerar_resumo(texto_limpo, "executivo")
        resumo_completo = self.gerar_resumo(texto_limpo, "completo")
        resumo_detalhado = self.gerar_resumo(texto_limpo, "detalhado")
        # 5.5. Gerar resumo personalizado, se solicitado
        resumo_personalizado = None
        if min_resumo is not None and max_resumo is not None:
            logger.info(f"Gerando resumo personalizado com tamanho min: {min_resumo}, max: {max_resumo}")
            resumo_personalizado = self.gerar_resumo(
                texto_limpo,
                tipo_resumo="personalizado",
                min_length=min_resumo,
                max_length=max_resumo
            )
        
        resumos = {
            'executivo': resumo_executivo,
            'completo': resumo_completo,
            'detalhado': resumo_detalhado,
            'personalizado': resumo_personalizado
        }
        # 6. Analisar sentimento
        analise_sentimento = self.analisar_sentimento(texto_limpo)
        
        # 7. Estatísticas básicas
        estatisticas = {
            'total_palavras': len(texto_limpo.split()),
            'total_caracteres': len(texto_limpo),
            'total_paragrafos': len(texto_limpo.split('\n')),
            'densidade_juridica': len(entidades['termos_juridicos']) / len(texto_limpo.split()) * 100
        }
        
        resultado = {
            'arquivo': os.path.basename(caminho_pdf),
            'tipo_documento': tipo_documento,
            'estatisticas': estatisticas,
            'entidades_juridicas': entidades,
            'resumos': {
                'executivo': resumo_executivo,
                'completo': resumo_completo,
                'detalhado': resumo_detalhado
            },
            'analise_sentimento': analise_sentimento,
            'texto_completo': texto_limpo[:2000] + '...' if len(texto_limpo) > 2000 else texto_limpo,
            'timestamp_analise': logger.handlers[0].formatter.formatTime(logger.makeRecord('', 0, '', 0, '', (), None))
        }
        
        logger.info("Análise completa finalizada!")
        return resultado



if __name__ == "__main__":
    # Instalar dependências necessárias:
    """
    pip install PyMuPDF pillow pytesseract torch transformers numpy
    
    Para OCR em português, instale o Tesseract:
    - Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-por
    - Windows: Baixar de https://github.com/UB-Mannheim/tesseract/wiki
    - macOS: brew install tesseract tesseract-lang
    """
    
    # exemplo_uso()