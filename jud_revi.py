#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador de PDFs Jurídicos com IA e OCR - Versão Aprimorada
Análise completa de documentos jurídicos usando modelos de IA gratuitos
Com suporte para resumos longos e cache
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
import hashlib
import json
from datetime import datetime
import pickle

warnings.filterwarnings("ignore")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFJuridicoAnalyzer:
    """
    Classe principal para análise de PDFs jurídicos com IA e OCR
    Versão aprimorada com resumos longos e cache
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, usar_cache: bool = True):
        """
        Inicializa o analisador
        
        Args:
            tesseract_path: Caminho para o executável do Tesseract (opcional)
            usar_cache: Se deve usar cache para economizar processamento
        """
        # Configurar Tesseract se caminho fornecido
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Configurar dispositivo (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Configurar cache
        self.usar_cache = usar_cache
        self.cache_dir = "cache_analises"
        if self.usar_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Inicializar modelos
        self._inicializar_modelos()
        
        # Configurações para documentos jurídicos
        self.termos_juridicos = [
            'sentença', 'acórdão', 'despacho', 'decisão', 'recurso',
            'apelação', 'embargos', 'agravo', 'mandado', 'habeas corpus',
            'ação', 'processo', 'autor', 'réu', 'juiz', 'desembargador',
            'tribunal', 'vara', 'comarca', 'foro', 'instância',
            'código civil', 'código penal', 'constituição', 'lei',
            'decreto', 'medida provisória', 'jurisprudência',
            'petição inicial', 'contestação', 'réplica', 'tréplica',
            'liminar', 'tutela', 'cautelar', 'mérito', 'prova',
            'testemunha', 'perícia', 'laudo', 'parecer', 'súmula'
        ]
        
    def _inicializar_modelos(self):
        """Inicializa os modelos de IA necessários - TODOS GRATUITOS"""
        try:
            logger.info("Carregando modelos de IA gratuitos...")
            
            # Modelo BERT português para análise
            self.tokenizer_bert = AutoTokenizer.from_pretrained(
                "neuralmind/bert-base-portuguese-cased"
            )
            self.model_bert = AutoModel.from_pretrained(
                "neuralmind/bert-base-portuguese-cased"
            ).to(self.device)
            
            # Tentar carregar o melhor modelo de resumo para português
            try:
                # Modelo T5 português (excelente para resumos)
                logger.info("Tentando carregar modelo T5 português...")
                self.summarizer = pipeline(
                    "summarization",
                    model="unicamp-dl/ptt5-base-portuguese-vocab",
                    tokenizer="unicamp-dl/ptt5-base-portuguese-vocab",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Modelo T5 português carregado com sucesso!")
            except:
                # Fallback para GPT2 português
                logger.info("Usando GPT2 português como fallback...")
                self.summarizer = pipeline(
                    "summarization",
                    model="pierreguillou/gpt2-small-portuguese",
                    tokenizer="pierreguillou/gpt2-small-portuguese",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            # Pipeline para análise de sentimentos multilíngue
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Modelo adicional para classificação de texto
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="joeddav/xlm-roberta-large-xnli",
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                self.classifier = None
                logger.warning("Classificador zero-shot não carregado")
            
            logger.info("Todos os modelos gratuitos carregados com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelos: {e}")
            self._carregar_modelos_minimos()
    
    def _carregar_modelos_minimos(self):
        """Carrega modelos mínimos em caso de erro"""
        logger.info("Carregando modelos mínimos de fallback...")
        try:
            # Modelo mínimo para resumos
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=0 if torch.cuda.is_available() else -1
            )
            # Análise de sentimento básica
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Erro crítico ao carregar modelos: {e}")
    
    def _get_cache_path(self, texto_hash: str, tipo_operacao: str) -> str:
        """Retorna o caminho do arquivo de cache"""
        return os.path.join(self.cache_dir, f"{texto_hash}_{tipo_operacao}.pkl")
    
    def _salvar_cache(self, texto_hash: str, tipo_operacao: str, dados: any):
        """Salva dados no cache"""
        if self.usar_cache:
            try:
                cache_path = self._get_cache_path(texto_hash, tipo_operacao)
                with open(cache_path, 'wb') as f:
                    pickle.dump(dados, f)
            except Exception as e:
                logger.warning(f"Erro ao salvar cache: {e}")
    
    def _carregar_cache(self, texto_hash: str, tipo_operacao: str) -> Optional[any]:
        """Carrega dados do cache se existir"""
        if self.usar_cache:
            try:
                cache_path = self._get_cache_path(texto_hash, tipo_operacao)
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
        return None
    
    def extrair_texto_pdf(self, caminho_pdf: str) -> str:
        """
        Extrai texto de um arquivo PDF com cache
        
        Args:
            caminho_pdf: Caminho para o arquivo PDF
            
        Returns:
            Texto extraído do PDF
        """
        # Verificar cache
        with open(caminho_pdf, 'rb') as f:
            pdf_hash = hashlib.md5(f.read()).hexdigest()
        
        cached_text = self._carregar_cache(pdf_hash, "texto_extraido")
        if cached_text:
            logger.info("Texto carregado do cache")
            return cached_text
        
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
            
            # Salvar no cache
            self._salvar_cache(pdf_hash, "texto_extraido", texto_completo)
            
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
            # Converter página para imagem com alta resolução
            matriz = fitz.Matrix(3.0, 3.0)  # Aumentar ainda mais a resolução
            pixmap = pagina.get_pixmap(matrix=matriz)
            img_data = pixmap.tobytes("png")
            
            # Carregar imagem com PIL
            imagem = Image.open(io.BytesIO(img_data))
            
            # Pré-processamento da imagem
            imagem = imagem.convert('L')  # Converter para escala de cinza
            
            # Configurar OCR para português com melhor precisão
            config_ocr = '--oem 3 --psm 6 -l por+eng'
            
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
        
        # Remover caracteres especiais desnecessários mas manter pontuação jurídica
        texto = re.sub(r'[^\w\s\.,;:!?()§°ºª\-\/]', '', texto)
        
        # Corrigir espaçamento em números de processo
        texto = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', texto)
        texto = re.sub(r'(\d)\s+\-\s+(\d)', r'\1-\2', texto)
        
        # Capitalizar início de sentenças
        texto = re.sub(r'([.!?]\s*)([a-z])', lambda m: m.group(1) + m.group(2).upper(), texto)
        
        return texto.strip()
    
    def dividir_texto_chunks(self, texto: str, tamanho_max: int = 1000) -> List[str]:
        """
        Divide texto em chunks inteligentes para processamento
        
        Args:
            texto: Texto completo
            tamanho_max: Tamanho máximo de cada chunk
            
        Returns:
            Lista de chunks de texto
        """
        # Tentar dividir por parágrafos primeiro
        paragrafos = texto.split('\n')
        chunks = []
        chunk_atual = []
        tamanho_atual = 0
        
        for paragrafo in paragrafos:
            tamanho_paragrafo = len(paragrafo)
            
            if tamanho_atual + tamanho_paragrafo <= tamanho_max:
                chunk_atual.append(paragrafo)
                tamanho_atual += tamanho_paragrafo
            else:
                if chunk_atual:
                    chunks.append('\n'.join(chunk_atual))
                
                # Se o parágrafo é muito grande, dividir por sentenças
                if tamanho_paragrafo > tamanho_max:
                    sentencas = re.split(r'[.!?]+', paragrafo)
                    sub_chunk = []
                    sub_tamanho = 0
                    
                    for sentenca in sentencas:
                        if sub_tamanho + len(sentenca) <= tamanho_max:
                            sub_chunk.append(sentenca)
                            sub_tamanho += len(sentenca)
                        else:
                            if sub_chunk:
                                chunks.append('. '.join(sub_chunk) + '.')
                            sub_chunk = [sentenca]
                            sub_tamanho = len(sentenca)
                    
                    if sub_chunk:
                        chunks.append('. '.join(sub_chunk) + '.')
                    
                    chunk_atual = []
                    tamanho_atual = 0
                else:
                    chunk_atual = [paragrafo]
                    tamanho_atual = tamanho_paragrafo
        
        if chunk_atual:
            chunks.append('\n'.join(chunk_atual))
        
        return chunks
    
    def extrair_entidades_juridicas(self, texto: str) -> Dict[str, List[str]]:
        """
        Extrai entidades jurídicas específicas do texto com melhorias
        
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
            'advogados': [],
            'tribunais': [],
            'varas': [],
            'datas_importantes': [],
            'valores_monetarios': [],
            'termos_juridicos': []
        }
        
        # Números de processo (padrões ampliados)
        padroes_processo = [
            r'\b\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4}\b',
            r'\b\d{4}\.\d{2}\.\d{2}\.\d{6}-\d{1}\b',
            r'\b\d{15}\b',
            r'Processo\s*n[ºo°]?\s*\d+[/\-]\d+'
        ]
        for padrao in padroes_processo:
            entidades['numeros_processo'].extend(re.findall(padrao, texto))
        
        # Leis citadas (padrão melhorado)
        padroes_lei = [
            r'Lei\s+n[ºo°]?\s*\d+\.?\d*/?-?\d*',
            r'Art\.?\s*\d+[º°]?(?:\s*,\s*\S+)?',
            r'Artigo\s+\d+[º°]?',
            r'CF/88|Código\s+(?:Civil|Penal|Processo\s+Civil|Processo\s+Penal)',
            r'CLT|CPC|CPP|CDC|ECA',
            r'§\s*\d+[º°]?'
        ]
        for padrao in padroes_lei:
            entidades['leis_citadas'].extend(re.findall(padrao, texto, re.IGNORECASE))
        
        # Advogados (OAB)
        padrao_oab = r'OAB[/\s]+[A-Z]{2}\s*n?[ºo°]?\s*\d+|OAB\s*:\s*\d+\s*-\s*[A-Z]{2}'
        entidades['advogados'] = re.findall(padrao_oab, texto, re.IGNORECASE)
        
        # Tribunais e Varas
        padrao_tribunal = r'TJ[A-Z]{2}|STF|STJ|TST|TSE|TRF\d?|TRT\d?|Tribunal\s+de\s+Justiça'
        entidades['tribunais'] = list(set(re.findall(padrao_tribunal, texto, re.IGNORECASE)))
        
        padrao_vara = r'\d+[ªº°]?\s*Vara\s*(?:Cível|Criminal|Federal|Trabalho|Família)'
        entidades['varas'] = re.findall(padrao_vara, texto, re.IGNORECASE)
        
        # Datas importantes
        padrao_data = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4}'
        entidades['datas_importantes'] = re.findall(padrao_data, texto)[:10]  # Limitar a 10
        
        # Valores monetários
        padrao_valor = r'R\$\s*[\d.,]+|(?:valor|montante|quantia)\s+de\s+R\$\s*[\d.,]+'
        entidades['valores_monetarios'] = re.findall(padrao_valor, texto, re.IGNORECASE)
        
        # Termos jurídicos específicos
        for termo in self.termos_juridicos:
            if termo.lower() in texto.lower():
                entidades['termos_juridicos'].append(termo)
        
        # Remover duplicatas
        for key in entidades:
            entidades[key] = list(set(entidades[key]))
        
        return entidades
    
    def gerar_resumo(self, texto: str, tipo_resumo: str = "completo", 
                     min_length: Optional[int] = None, max_length: Optional[int] = None) -> str:
        """
        Gera resumo do texto usando IA com suporte para resumos longos
        
        Args:
            texto: Texto para resumir
            tipo_resumo: Tipo de resumo ("executivo", "completo", "detalhado", "ultra_detalhado")
            min_length: Tamanho mínimo customizado
            max_length: Tamanho máximo customizado
            
        Returns:
            Resumo gerado
        """
        # Criar hash para cache
        texto_hash = hashlib.md5(f"{texto}{tipo_resumo}{min_length}{max_length}".encode()).hexdigest()
        
        # Verificar cache
        resumo_cached = self._carregar_cache(texto_hash, f"resumo_{tipo_resumo}")
        if resumo_cached:
            logger.info(f"Resumo '{tipo_resumo}' carregado do cache")
            return resumo_cached
        
        try:
            # Definir tamanhos de resumo
            if min_length is not None and max_length is not None:
                final_max_length = max_length
                final_min_length = min_length
            else:
                if tipo_resumo == "executivo":
                    final_max_length = 150
                    final_min_length = 75
                elif tipo_resumo == "detalhado":
                    final_max_length = 500
                    final_min_length = 250
                elif tipo_resumo == "ultra_detalhado":
                    final_max_length = 1000
                    final_min_length = 500
                else:  # completo
                    final_max_length = 350
                    final_min_length = 175
            
            # Dividir texto em chunks apropriados
            tamanho_chunk = 1500 if tipo_resumo in ["ultra_detalhado", "detalhado"] else 1000
            chunks = self.dividir_texto_chunks(texto, tamanho_chunk)
            resumos_parciais = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Resumindo chunk {i+1}/{len(chunks)} para resumo '{tipo_resumo}'")
                
                try:
                    # Calcular tamanho proporcional para cada chunk
                    max_length_chunk = min(512, final_max_length // max(1, len(chunks)))
                    min_length_chunk = max_length_chunk // 2
                    
                    resumo = self.summarizer(
                        chunk,
                        max_length=max_length_chunk,
                        min_length=min_length_chunk,
                        do_sample=False
                    )
                    resumos_parciais.append(resumo[0]['summary_text'])
                except Exception as e:
                    logger.warning(f"Erro ao resumir chunk {i+1}: {e}")
                    # Fallback mais robusto
                    sentencas = chunk.split('.')
                    num_sentencas = max(5, len(sentencas) // 4)  # Pegar 25% das sentenças
                    resumos_parciais.append('. '.join(sentencas[:num_sentencas]) + '.')
            
            resumo_final = ' '.join(resumos_parciais)
            
            # Se resumo ainda muito longo e não for ultra_detalhado, resumir novamente
            if len(resumo_final.split()) > final_max_length * 2 and tipo_resumo != "ultra_detalhado":
                try:
                    logger.info("Realizando segunda passada de resumo...")
                    resumo_final = self.summarizer(
                        resumo_final,
                        max_length=final_max_length,
                        min_length=final_min_length,
                        do_sample=False
                    )[0]['summary_text']
                except:
                    pass
            
            # Salvar no cache
            self._salvar_cache(texto_hash, f"resumo_{tipo_resumo}", resumo_final)
            
            return resumo_final
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {e}")
            return self._resumo_manual_aprimorado(texto, tipo_resumo)
    
    def gerar_resumo_extenso(self, texto: str, percentual: float = 0.4) -> str:
        """
        Gera um resumo que mantém X% do texto original
        
        Args:
            texto: Texto para resumir
            percentual: Percentual do texto a manter (0.4 = 40%)
        
        Returns:
            Resumo extenso
        """
        texto_hash = hashlib.md5(f"{texto}{percentual}".encode()).hexdigest()
        
        # Verificar cache
        resumo_cached = self._carregar_cache(texto_hash, f"resumo_percentual_{percentual}")
        if resumo_cached:
            return resumo_cached
        
        try:
            # Calcular tamanho alvo
            palavras_originais = len(texto.split())
            tamanho_alvo = int(palavras_originais * percentual)
            
            # Dividir em chunks grandes
            chunks = self.dividir_texto_chunks(texto, 2000)
            resumos_parciais = []
            
            # Calcular tamanho por chunk
            palavras_por_chunk = max(100, tamanho_alvo // len(chunks))
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processando chunk {i+1}/{len(chunks)} - Mantendo {percentual*100}% do conteúdo")
                
                try:
                    # Limitar ao máximo suportado pelo modelo
                    max_length = min(palavras_por_chunk, 512)
                    min_length = max_length // 2
                    
                    resumo = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    resumos_parciais.append(resumo[0]['summary_text'])
                except Exception as e:
                    # Fallback: manter percentual das sentenças
                    sentencas = chunk.split('.')
                    num_manter = int(len(sentencas) * percentual)
                    resumos_parciais.append('. '.join(sentencas[:num_manter]) + '.')
            
            resumo_final = ' '.join(resumos_parciais)
            
            # Salvar no cache
            self._salvar_cache(texto_hash, f"resumo_percentual_{percentual}", resumo_final)
            
            return resumo_final
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo extenso: {e}")
            return self._resumo_manual_aprimorado(texto, "extenso")
    
    def _resumo_manual_aprimorado(self, texto: str, tipo: str = "completo") -> str:
        """
        Gera resumo manual aprimorado em caso de falha da IA
        
        Args:
            texto: Texto para resumir
            tipo: Tipo de resumo desejado
            
        Returns:
            Resumo manual
        """
        paragrafos = texto.split('\n')
        sentencas_importantes = []
        
        # Definir quantas sentenças pegar baseado no tipo
        proporcoes = {
            "executivo": 0.1,
            "completo": 0.2,
            "detalhado": 0.3,
            "ultra_detalhado": 0.4,
            "extenso": 0.5
        }
        proporcao = proporcoes.get(tipo, 0.2)
        
        # Scoring de sentenças baseado em termos jurídicos
        for paragrafo in paragrafos:
            if len(paragrafo) > 50:
                sentencas = paragrafo.split('.')
                for sentenca in sentencas:
                    if len(sentenca) > 20:
                        # Calcular score baseado em termos jurídicos
                        score = sum(1 for termo in self.termos_juridicos if termo.lower() in sentenca.lower())
                        if score > 0:
                            sentencas_importantes.append((score, sentenca))
        
        # Ordenar por importância e pegar as top N
        sentencas_importantes.sort(key=lambda x: x[0], reverse=True)
        num_sentencas = int(len(sentencas_importantes) * proporcao)
        
        resumo = []
        for _, sentenca in sentencas_importantes[:num_sentencas]:
            resumo.append(sentenca.strip() + '.')
        
        return ' '.join(resumo)
    
    def analisar_sentimento(self, texto: str) -> Dict[str, any]:
        """
        Analisa o sentimento/tom do documento com melhorias
        
        Args:
            texto: Texto para análise
            
        Returns:
            Resultado da análise de sentimento
        """
        try:
            chunks = self.dividir_texto_chunks(texto, 500)
            resultados_detalhados = []
            
            for i, chunk in enumerate(chunks[:10]):  # Analisar até 10 chunks
                try:
                    resultado = self.sentiment_analyzer(chunk)
                    # Converter score de 1-5 estrelas para sentimento
                    if 'label' in resultado[0]:
                        label = resultado[0]['label']
                        if '5' in label or '4' in label:
                            sentimento = 'positivo'
                        elif '1' in label or '2' in label:
                            sentimento = 'negativo'
                        else:
                            sentimento = 'neutro'
                    else:
                        sentimento = resultado[0].get('label', 'neutro').lower()
                    
                    resultados_detalhados.append({
                        'chunk': i + 1,
                        'sentimento': sentimento,
                        'confianca': resultado[0]['score']
                    })
                except:
                    continue
            
            # Análise agregada
            sentimentos_count = {'positivo': 0, 'negativo': 0, 'neutro': 0}
            confiancas = {'positivo': [], 'negativo': [], 'neutro': []}
            
            for res in resultados_detalhados:
                sent = res['sentimento']
                if sent in sentimentos_count:
                    sentimentos_count[sent] += 1
                    confiancas[sent].append(res['confianca'])
            
            # Determinar tom geral
            tom_geral = max(sentimentos_count, key=sentimentos_count.get)
            
            return {
                'tom_geral': tom_geral,
                'distribuicao': sentimentos_count,
                'confianca_media': {
                    k: np.mean(v) if v else 0 for k, v in confiancas.items()
                },
                'analise_detalhada': resultados_detalhados[:5]  # Primeiros 5 chunks
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimento: {e}")
            return {'tom_geral': 'indeterminado', 'erro': str(e)}
    
    def classificar_documento(self, texto: str, entidades: Dict) -> Dict[str, any]:
        """
        Classifica o tipo de documento jurídico com mais detalhes
        
        Args:
            texto: Texto do documento
            entidades: Entidades extraídas
            
        Returns:
            Classificação detalhada do documento
        """
        texto_lower = texto.lower()
        
        # Padrões expandidos para diferentes tipos de documentos
        padroes = {
            'Sentença': {
                'palavras': ['sentença', 'julgo', 'dispositivo', 'condeno', 'absolvo', 'procedente', 'improcedente'],
                'peso': 3
            },
            'Acórdão': {
                'palavras': ['acórdão', 'acordam', 'turma', 'relator', 'revisor', 'voto'],
                'peso': 3
            },
            'Despacho': {
                'palavras': ['despacho', 'intime-se', 'cumpra-se', 'cite-se', 'vista'],
                'peso': 2
            },
            'Petição Inicial': {
                'palavras': ['petição inicial', 'requer', 'preliminarmente', 'autor', 'ação'],
                'peso': 3
            },
            'Contestação': {
                'palavras': ['contestação', 'contesta', 'réu', 'impugna', 'defesa'],
                'peso': 3
            },
            'Recurso': {
                'palavras': ['recurso', 'apelação', 'agravo', 'embargos', 'recorrente'],
                'peso': 3
            },
            'Contrato': {
                'palavras': ['contrato', 'partes', 'cláusula', 'objeto', 'obrigações', 'contratante'],
                'peso': 2
            },
            'Parecer': {
                'palavras': ['parecer', 'opina', 'manifestação', 'entendimento', 'análise jurídica'],
                'peso': 2
            }
        }
        
        pontuacoes = {}
        for tipo, config in padroes.items():
            pontuacao = 0
            for palavra in config['palavras']:
                ocorrencias = texto_lower.count(palavra)
                pontuacao += ocorrencias * config['peso']
            pontuacoes[tipo] = pontuacao
        
        # Determinar tipo principal
        tipo_principal = max(pontuacoes, key=pontuacoes.get) if max(pontuacoes.values()) > 0 else 'Documento Jurídico'
        
        # Calcular confiança
        total_pontos = sum(pontuacoes.values())
        confianca = pontuacoes[tipo_principal] / total_pontos if total_pontos > 0 else 0
        
        # Usar classificador zero-shot se disponível
        classificacao_ml = None
        if self.classifier:
            try:
                candidatos = list(padroes.keys())
                resultado = self.classifier(texto[:1000], candidate_labels=candidatos)
                classificacao_ml = {
                    'tipo': resultado['labels'][0],
                    'confianca': resultado['scores'][0]
                }
            except:
                pass
        
        return {
            'tipo_principal': tipo_principal,
            'confianca': confianca,
            'pontuacoes_detalhadas': pontuacoes,
            'classificacao_ml': classificacao_ml,
            'subtipo': self._identificar_subtipo(texto, tipo_principal)
        }
    
    def _identificar_subtipo(self, texto: str, tipo_principal: str) -> Optional[str]:
        """Identifica subtipos específicos do documento"""
        texto_lower = texto.lower()
        
        subtipos = {
            'Sentença': {
                'condenatória': ['condeno', 'pena', 'condenação'],
                'absolutória': ['absolvo', 'absolvição', 'inocente'],
                'homologatória': ['homologo', 'acordo', 'transação']
            },
            'Recurso': {
                'apelação': ['apelação', 'apelante'],
                'agravo de instrumento': ['agravo de instrumento'],
                'embargos de declaração': ['embargos de declaração', 'omissão', 'contradição']
            }
        }
        
        if tipo_principal in subtipos:
            for subtipo, palavras in subtipos[tipo_principal].items():
                if any(palavra in texto_lower for palavra in palavras):
                    return subtipo
        
        return None
    
    def analisar_pdf_completo(self, caminho_pdf: str, 
                              min_resumo: Optional[int] = None, 
                              max_resumo: Optional[int] = None,
                              gerar_resumo_percentual: Optional[float] = None) -> Dict[str, any]:
        """
        Realiza análise completa do PDF com todas as melhorias
        
        Args:
            caminho_pdf: Caminho para o arquivo PDF
            min_resumo: Tamanho mínimo para resumo personalizado
            max_resumo: Tamanho máximo para resumo personalizado
            gerar_resumo_percentual: Percentual do texto para resumo extenso (ex: 0.3 para 30%)
            
        Returns:
            Dicionário com análise completa
        """
        logger.info(f"Iniciando análise completa do PDF: {caminho_pdf}")
        inicio = datetime.now()
        
        # 1. Extrair texto (com cache)
        texto_bruto = self.extrair_texto_pdf(caminho_pdf)
        if not texto_bruto:
            return {'erro': 'Não foi possível extrair texto do PDF'}
        
        # 2. Preprocessar texto
        texto_limpo = self.preprocessar_texto(texto_bruto)
        
        # 3. Extrair entidades jurídicas
        logger.info("Extraindo entidades jurídicas...")
        entidades = self.extrair_entidades_juridicas(texto_limpo)
        
        # 4. Classificar documento
        logger.info("Classificando documento...")
        classificacao = self.classificar_documento(texto_limpo, entidades)
        
        # 5. Gerar todos os resumos
        logger.info("Gerando resumos...")
        resumos = {
            'executivo': self.gerar_resumo(texto_limpo, "executivo"),
            'completo': self.gerar_resumo(texto_limpo, "completo"),
            'detalhado': self.gerar_resumo(texto_limpo, "detalhado"),
            'ultra_detalhado': self.gerar_resumo(texto_limpo, "ultra_detalhado")
        }
        
        # 5.1 Resumo personalizado se solicitado
        if min_resumo and max_resumo:
            logger.info(f"Gerando resumo personalizado (min: {min_resumo}, max: {max_resumo})...")
            resumos['personalizado'] = self.gerar_resumo(
                texto_limpo, 
                tipo_resumo="personalizado",
                min_length=min_resumo,
                max_length=max_resumo
            )
        
        # 5.2 Resumo percentual se solicitado
        if gerar_resumo_percentual:
            logger.info(f"Gerando resumo de {gerar_resumo_percentual*100}% do texto...")
            resumos['percentual'] = self.gerar_resumo_extenso(texto_limpo, gerar_resumo_percentual)
        
        # 6. Analisar sentimento
        logger.info("Analisando sentimento do documento...")
        analise_sentimento = self.analisar_sentimento(texto_limpo)
        
        # 7. Estatísticas detalhadas
        palavras = texto_limpo.split()
        estatisticas = {
            'total_palavras': len(palavras),
            'total_caracteres': len(texto_limpo),
            'total_paragrafos': len(texto_limpo.split('\n')),
            'total_paginas': texto_bruto.count('--- PÁGINA'),
            'densidade_juridica': len(entidades['termos_juridicos']) / len(palavras) * 100,
            'complexidade': self._calcular_complexidade(texto_limpo, entidades),
            'tempo_leitura_estimado': f"{len(palavras) // 200} minutos"
        }
        
        # 8. Análise temporal
        analise_temporal = self._analisar_linha_tempo(entidades.get('datas_importantes', []))
        
        # 9. Compilar resultado final
        resultado = {
            'arquivo': os.path.basename(caminho_pdf),
            'classificacao_documento': classificacao,
            'estatisticas': estatisticas,
            'entidades_juridicas': entidades,
            'resumos': resumos,
            'analise_sentimento': analise_sentimento,
            'analise_temporal': analise_temporal,
            'metadados': {
                'timestamp_analise': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tempo_processamento': str(datetime.now() - inicio),
                'versao_analisador': '2.0',
                'modelos_utilizados': {
                    'resumo': 'ptt5-base-portuguese / gpt2-portuguese',
                    'sentimento': 'bert-multilingual-sentiment',
                    'classificacao': 'xlm-roberta-xnli'
                }
            },
            'texto_preview': texto_limpo[:1000] + '...' if len(texto_limpo) > 1000 else texto_limpo
        }
        
        logger.info(f"Análise completa finalizada em {datetime.now() - inicio}")
        return resultado
    
    def _calcular_complexidade(self, texto: str, entidades: Dict) -> str:
        """Calcula a complexidade do documento"""
        palavras = texto.split()
        
        # Fatores de complexidade
        media_palavras_sentenca = len(palavras) / max(1, texto.count('.'))
        num_leis_citadas = len(entidades.get('leis_citadas', []))
        num_termos_tecnicos = len(entidades.get('termos_juridicos', []))
        
        score = 0
        if media_palavras_sentenca > 25:
            score += 2
        elif media_palavras_sentenca > 15:
            score += 1
            
        if num_leis_citadas > 10:
            score += 2
        elif num_leis_citadas > 5:
            score += 1
            
        if num_termos_tecnicos > 20:
            score += 2
        elif num_termos_tecnicos > 10:
            score += 1
        
        if score >= 5:
            return "Alta"
        elif score >= 3:
            return "Média"
        else:
            return "Baixa"
    
    def _analisar_linha_tempo(self, datas: List[str]) -> Dict[str, any]:
        """Analisa a linha temporal do documento"""
        if not datas:
            return {'possui_datas': False}
        
        try:
            # Converter datas para formato padrão
            datas_parseadas = []
            for data in datas:
                # Tentar diferentes formatos
                for formato in ['%d/%m/%Y', '%d-%m-%Y', '%d de %B de %Y']:
                    try:
                        dt = datetime.strptime(data, formato)
                        datas_parseadas.append(dt)
                        break
                    except:
                        continue
            
            if datas_parseadas:
                datas_parseadas.sort()
                return {
                    'possui_datas': True,
                    'data_mais_antiga': datas_parseadas[0].strftime('%d/%m/%Y'),
                    'data_mais_recente': datas_parseadas[-1].strftime('%d/%m/%Y'),
                    'periodo_dias': (datas_parseadas[-1] - datas_parseadas[0]).days,
                    'total_datas': len(datas_parseadas)
                }
        except:
            pass
        
        return {
            'possui_datas': True,
            'total_datas_encontradas': len(datas),
            'observacao': 'Não foi possível processar todas as datas'
        }


# Exemplo de uso
if __name__ == "__main__":
    # Criar analisador
    analyzer = PDFJuridicoAnalyzer()
    
    # Analisar PDF com todas as opções
    resultado = analyzer.analisar_pdf_completo(
        "documento_juridico.pdf",
        min_resumo=800,           # Resumo personalizado de 800-1500 palavras
        max_resumo=1500,
        gerar_resumo_percentual=0.3  # Resumo com 30% do texto original
    )
    
    # Exibir resultados
    print(json.dumps(resultado, indent=2, ensure_ascii=False))