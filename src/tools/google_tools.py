from typing import Optional, List, Dict, Any
import google.genai as genai
from google.genai import types
from langchain.tools import tool
from config.config import Config

config = Config()


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=config.GOOGLE_API_KEY)


@tool
def google_grounding_search(text: str, url_list: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Use esta tool para busca na web com grounding da Google e obter uma resposta objetiva
    com base em fontes atuais da internet.

    Quando usar:
    - Perguntas que exigem dados recentes, verificações ou buscas na web.
    - Confirmação rápida de informações externas ao grafo Neo4j.

    Quando não usar:
    - Consultas estruturadas sobre empresas ja cobertas pelo Neo4j (prefira `neo4j_*`).

    Entrada:
    - `text` (str): pergunta completa e especifica em linguagem natural.

    Saida (dict):
    - `grounding_search_result.answer`: resposta textual gerada com grounding web.
    - `grounding_search_result.sources`: lista de URLs indexadas na busca.
    - `grounding_search_result.search_queries`: consultas enviadas ao Google.
    """
    client = get_genai_client()
    urls = []
    queries = []
    if url_list and len(url_list):
        grounding_tool = types.Tool(url_context=types.UrlContext())
        content_config = types.GenerateContentConfig(tools=[grounding_tool])
        url_list_str = ", ".join(url_list)
        response = client.models.generate_content(
            model=config.GOOGLE_TOOL_MODEL,
            contents=f"Realize a seguinte consulta: {text}.\nUse somente as seguintes URLs como fontes de busca: {url_list_str}",
            config=content_config,
        )
        answer = ""
        for each in response.candidates[0].content.parts:
            answer += each.text + "\n"

        metadata = response.candidates[0].url_context_metadata if response.candidates else None
        if metadata:
            urls = [
                {"uri": entry.retrieved_url, "title": ""}
                for entry in (metadata.url_metadata or [])
                if getattr(entry, "retrieved_url", None)
            ]
    else:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        content_config = types.GenerateContentConfig(tools=[grounding_tool])
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=text,
            config=content_config,
        )
        answer = response.text

        metadata = response.candidates[0].grounding_metadata if response.candidates else None
        if metadata:
            urls = [
                {"uri": chunk.web.uri, "title": chunk.web.title}
                for chunk in (metadata.grounding_chunks or [])
                if chunk.web
            ]
            queries = list(metadata.web_search_queries or [])

    return {
        "grounding_search_result": {
            "answer": answer,
            "sources": urls,
            "search_queries": queries,
        }
    }