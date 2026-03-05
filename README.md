# poc-skills

Agente conversacional em Python com **LangGraph + LangChain**, usando **Gemini** como LLM principal e ferramentas para:
- execução de código Python em sandbox (E2B)
- memória temporal e consultas em grafo com Graphiti + Neo4j
- integração com API da Anthropic (implementada no projeto, hoje não ativada por padrão no agente)

O ponto de entrada da aplicação é: `src/main.py`.

## Arquitetura

- `src/main.py`: inicializa o `Agent` e inicia o loop interativo.
- `src/agent/agent.py`: orquestração com `StateGraph` (nó de raciocínio + nó de execução de tools).
- `src/tools/e2b.py`: tool `e2b_run_code` para executar Python em sandbox.
- `src/tools/graphiti.py`: tools de eventos temporais:
  - `graphiti_add_event`
  - `graphiti_remove_event`
  - `graphiti_get_entity_edges`
  - `graphiti_list_recent_episodes`
  - `graphiti_search_events`
- `src/tools/anthropic.py`: tools para upload/download e chamadas ao Claude via API da Anthropic.
- `src/config/*`: configuração de ambiente, logging e parâmetros da Anthropic.
- `docker-compose.yml`: stack local do Neo4j para Graphiti.

## Requisitos

- Python `>= 3.13`
- Docker + Docker Compose (para Neo4j)
- Chaves de API:
  - Google (`GOOGLE_API_KEY`) para Gemini
  - E2B (`E2B_API_KEY`) para sandbox de código
  - OpenAI (`OPENAI_API_KEY`) para embeddings usados pelo Graphiti
  - Anthropic (`ANTHROPIC_API_KEY`) se quiser usar tools da Anthropic

## Instalação

### Opção 1: usando `uv` (recomendado)

```bash
uv sync
```

### Opção 2: venv + pip

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Configuração

Crie/edite o `.env` na raiz com pelo menos:

```env
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro

E2B_API_KEY=...

OPENAI_API_KEY=...

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphiti123

# opcionais (Anthropic)
ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-sonnet-4-5-20250929
```

## Subindo o Neo4j

```bash
docker compose up -d
```

Serviços padrão:
- Neo4j Browser: `http://localhost:7474`
- Bolt: `localhost:7687`

## Executando o projeto

Com ambiente configurado, execute:

```bash
python src/main.py
```

ou com `uv`:

```bash
uv run python src/main.py
```

Você verá um prompt interativo:
- `/status` mostra estado atual (sessão, arquivos, etc.)
- `/quit` ou `/exit` encerra

## Como o agente funciona

O fluxo no `Agent` usa dois nós:

1. **agent node**  
   Recebe histórico de mensagens, gera próxima ação com `ChatGoogleGenerativeAI` e decide chamadas de tools.

2. **tool node**  
   Resolve a tool pelo nome a partir de `self.tools`, executa com `tool.ainvoke(...)`, registra resultado e devolve `ToolMessage` ao grafo.

Isso cria um loop de raciocínio -> ação -> observação até finalizar a resposta.

## Tools disponíveis hoje no agente

Atualmente `self.tools` está configurado com:
- `e2b_run_code`
- `graphiti_add_event`
- `graphiti_remove_event`
- `graphiti_get_entity_edges`
- `graphiti_search_events`
- `graphiti_list_recent_episodes`

As tools da Anthropic existem no código, mas estão comentadas na lista principal do agente.

## Exemplos de uso no prompt interativo

- “Registre um evento: reunião com Maria às 14:00 sobre roadmap.”
- “Liste os últimos 5 episódios.”
- “Busque eventos depois de 2026-03-01T00:00:00.”
- “Remova o episódio com UUID `<uuid>`.”
- “Execute um código Python que calcule média e desvio padrão de [1,2,3,4,5].”

## Observações

- O Graphiti inicializa índices/constraints no Neo4j na primeira execução.
- `graphiti_search_events` aplica filtro temporal e textual sobre episódios recuperados.
- A tool de remoção de episódio depende do `episode_uuid` (retornado em listagens/buscas).

## Troubleshooting

- Erro de conexão Neo4j: verifique `docker compose ps` e credenciais do `.env`.
- Erro de autenticação Gemini/E2B/OpenAI: confirme variáveis e chaves.
- Nenhuma tool executada: valide se o modelo está chamando tools e se elas estão em `self.tools`.

## Estrutura do projeto

```text
src/
  agent/
    agent.py
  config/
    anthropic.py
    config.py
    logger.py
  tools/
    anthropic.py
    e2b.py
    graphiti.py
  utils/
    http.py
  main.py
docker-compose.yml
pyproject.toml
```
