# Reto Unidad 4: GAIA

## Resultado

Resolvimos la primera unidad y desarrollamos un agente para afrontar el reto GAIA de la Unidad 4. El agente obtuvo un **60 % de aciertos** en la evaluación realizada, omitiendo las preguntas que dependían de videos o imágenes.

Para mantener el agente operativo frente a problemas de disponibilidad y compatibilidad, configuramos una cadena con distintas versiones de Gemini. Si un modelo falla o no responde, el código intenta continuar con otro modelo de la cadena.

La clave de Gemini es un secreto y no se incluye en este repositorio. El agente funciona al configurar `GEMINI_API_KEY` en el entorno local; el archivo `.env.example` solo muestra nombres de variables y valores de ejemplo no secretos.

## Herramientas del agente

| Herramienta | Uso |
|---|---|
| `web_search` | Busca información en la web con DuckDuckGo y usa Wikipedia como alternativa si la búsqueda falla. |
| `wikipedia_page` | Recupera artículos de Wikipedia en inglés, con tablas y opción de extraer una sección por palabra clave. |
| `visit_webpage` | Abre páginas web y devuelve su contenido en texto/Markdown. |
| `read_file` | Lee archivos adjuntos: Excel, CSV/TSV, PDF y formatos de texto como TXT, Markdown, JSON o código. |
| `final_answer` | Entrega la respuesta final en el formato conciso que requiere GAIA. |
| Python del agente | Realiza cálculos y procesa datos; se autorizan módulos como `pandas`, `numpy`, `requests`, `json` y `math`. |

Las imágenes pueden adjuntarse al modelo para visión, pero en la evaluación reportada se omitieron las preguntas de imagen. El agente no procesa audio o video; esas preguntas se omiten por defecto para no gastar llamadas del modelo.

## Configuración

Se requiere Python 3.11. Instala las dependencias con:

```bash
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y configura `GEMINI_API_KEY` con una clave propia. Para enviar resultados al sistema de evaluación también configura `HF_USERNAME` y, si corresponde, `AGENT_CODE_URL`. No publiques `.env` ni la clave.

## Ejecución

```bash
python app.py                         # Interfaz de chat local
python app.py --gaia --limit 2        # Prueba con dos preguntas
python app.py --gaia                  # Ejecuta el reto y guarda/reanuda respuestas
python app.py --gaia --submit         # Ejecuta lo pendiente y envía resultados
python app.py --submit-only           # Envía respuestas ya guardadas
python app.py --ping                  # Comprueba la cadena de modelos configurada
```

Las respuestas se guardan en `answers.json` y las preguntas descargadas en `questions.json`. El tiempo total de ejecución no se documenta porque no se encontró una medición registrada.
