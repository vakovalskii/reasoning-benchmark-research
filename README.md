# LLM Reasoning Benchmark Research 🧠

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://vakovalskii.github.io/reasoning-benchmark-research/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Comprehensive study of 5 reasoning approaches on SQuAD dataset**

Набор тестов для оценки производительности языковых моделей на датасете SQuAD v1.1 с фокусом на влияние reasoning (рассуждений) на точность и скорость.

📖 **[Read Full Research on GitHub Pages →](https://vakovalskii.github.io/reasoning-benchmark-research/)**

---

## 🚀 Quick Start

```bash
# Clone repository
git clone git@github.com:vakovalskii/reasoning-benchmark-research.git
cd reasoning-benchmark-research

# Install dependencies
pip install -r requirements.txt

# Configure API credentials
cp config.py.example config.py
# Edit config.py with your API key and endpoint

# Download SQuAD dataset
python download_squad.py

# Run benchmarks
cd without_reasoning && python test_squad_without_reasoning.py
cd ../with_structured_output && python test_squad_with_so.py
cd ../with_two_step_so && python test_squad_two_step_so.py
cd ../with_react && python test_squad_with_reasoning.py
cd ../with_react_two_tools && python test_squad_react_two_tools.py
```

---

## 🎯 Цель исследования

Сравнить различные подходы к использованию **reasoning (рассуждений)** в языковых моделях для задачи извлечения ответов из контекста. Мы тестируем две ключевые гипотезы:

1. **Явное рассуждение перед ответом улучшает точность**
2. **Способ получения финального ответа влияет на результат**

### Тестируемые подходы:

1. **Without Reasoning (Baseline)**
   - Модель сразу отвечает без промежуточных рассуждений
   - Самый быстрый подход
   - Базовая линия для сравнения

2. **Single-Step Structured Output (SO)**
   - Модель генерирует `reasoning` и `answer` одновременно через JSON schema
   - Рассуждения и ответ в одном запросе
   - **Проверяем:** помогает ли SO структурировать мышление?
   - **Проверяем:** мешает ли одновременная генерация reasoning + answer?

3. **Two-Step Structured Output**
   - Шаг 1: Генерация `reasoning` через SO
   - Шаг 2: Reasoning вставляется в контекст (assistant role), модель отвечает **свободным текстом**
   - **Проверяем:** помогает ли разделение на два шага?
   - **Проверяем:** лучше ли free-form answer, чем SO answer?

4. **ReAct (Function Calling - Single Tool)**
   - Шаг 1: Модель вызывает функцию `generate_reasoning`
   - Шаг 2: На основе результата функции модель извлекает ответ **свободным текстом**
   - **Проверяем:** дает ли tool-based reasoning лучшие результаты?
   - **Проверяем:** влияет ли агентный подход на качество?

5. **ReAct Two Tools (Function Calling - Two Tools)**
   - Шаг 1: Модель вызывает `generate_reasoning(reasoning: str)`
   - Шаг 2: Модель вызывает `submit_answer(answer: str)`
   - **Проверяем:** лучше ли извлекать answer через tool call, чем free-form?
   - **Проверяем:** влияет ли явная структура на точность?

### Ключевые вопросы исследования:

**О reasoning:**
- ❓ **Улучшает ли reasoning точность?** На сколько процентов?
- ❓ **Какой способ reasoning лучше?** SO vs Function Calling?

**О способе получения ответа:**
- ❓ **Free-form vs Structured Output для answer?** Что точнее?
- ❓ **Одновременная генерация vs разделение?** Мешает ли SO при генерации answer?
- ❓ **Tool call для answer vs free-form?** Дает ли явная структура преимущество?

**О trade-offs:**
- ❓ **Какой ценой?** Как меняется скорость и латентность?
- ❓ **Стоит ли оно того?** Оправдывает ли прирост точности потерю скорости?

---

## 📋 Описание задачи

**SQuAD (Stanford Question Answering Dataset)** - это задача **извлечения ответа из контекста** (Reading Comprehension).

### Что должна делать модель:

**Дано:**
- **Контекст** - параграф текста
- **Вопрос** - вопрос по этому контексту

**Требуется:**
- Найти точный ответ в контексте
- Ответ должен быть **дословной цитатой** из текста (не перефразировать!)

### Пример:

**Контекст:**
```
The Panthers used the San Jose State practice facility and stayed at the 
San Jose Marriott. The Broncos practiced at Stanford University and stayed 
at the Santa Clara Marriott.
```

**Вопрос:**
```
Which hotel did the Broncos use for Super Bowl 50?
```

**Правильный ответ:**
```
Santa Clara Marriott
```

### Что требуется от модели:

1. **Понять вопрос** - что именно спрашивают
2. **Найти релевантную часть контекста** - где упоминаются Broncos и отель
3. **Извлечь точный ответ** - "Santa Clara Marriott" (дословно из текста)
4. **НЕ генерировать** - ответ должен быть точной цитатой

---

## 💬 Промпты (намеренно простые)

**Важно:** Все промпты максимально простые и **не содержат специальных инструкций** для задачи SQuAD. Мы тестируем способность модели понимать задачу из контекста, а не следовать детальным инструкциям.

### 1. Without Reasoning

**System:**
```
You are a helpful assistant. Answer the question based on the given context.
```

**User:**
```
Context: {context}

Question: {question}

Answer:
```

### 2. With Structured Output (Single-Step)

**System:**
```
You are a helpful assistant. Answer questions based on the given context.
```

**User:**
```
Context: {context}

Question: {question}

Think step-by-step, then provide the exact answer from the context.
```

**JSON Schema:**
```json
{
  "reasoning": "Step-by-step analysis of where the answer is in the context",
  "answer": "The exact answer extracted from the context"
}
```

### 3. With Two-Step SO

**Step 1 - System:**
```
You are a helpful assistant. Analyze the context to find where the answer is located.
```

**Step 1 - User:**
```
Context: {context}

Question: {question}

Think step-by-step about where the answer is in the context.
```

**Step 2 - Assistant:** `{reasoning from step 1}`

**Step 2 - User:**
```
Based on your analysis, extract the exact answer from the context.
```

### 4. With ReAct (Function Calling)

**System:**
```
You are a helpful assistant. Use the generate_reasoning tool to think step-by-step, 
then provide the final answer.
```

**User:**
```
Context: {context}

Question: {question}

First, use the generate_reasoning tool to analyze where the answer is in the context. 
Then extract the exact answer.
```

**Tool Definition:**
```json
{
  "name": "generate_reasoning",
  "description": "Generate step-by-step reasoning to find the answer in the context",
  "parameters": {
    "reasoning": "Step-by-step analysis of where the answer is in the context"
  }
}
```

**Ключевое наблюдение:** Промпты не содержат:
- ❌ Упоминания "SQuAD"
- ❌ Детальных инструкций по извлечению ответов
- ❌ Примеров правильных ответов
- ❌ Специальных форматирований
- ✅ Только базовые инструкции: "answer based on context" и "extract exact answer"

---

## Структура проекта

```
reasoning_benchmark/
├── config.py                          # Единый конфиг для всех тестов
├── squad_data/                        # Данные SQuAD
│   ├── train.json
│   └── validation.json
├── without_reasoning/                 # Тесты без reasoning
│   ├── test_squad_without_reasoning.py
│   └── results/                       # Результаты тестов
├── with_structured_output/            # Тесты с SO (1 шаг: reasoning + answer)
│   ├── test_squad_with_so.py
│   └── results/                       # Результаты тестов
├── with_two_step_so/                  # Тесты с Two-Step SO
│   ├── test_squad_two_step_so.py
│   └── results/                       # Результаты тестов
└── with_react/                        # Тесты с ReAct (Function Calling)
    ├── test_squad_with_reasoning.py
    └── results/                       # Результаты тестов
```

## Конфигурация

Все параметры настраиваются в файле `config.py`:

- **API Configuration**: `API_KEY`, `BASE_URL`, `MODEL`
- **Model Parameters**: `TEMPERATURE`, `MAX_TOKENS_*`
- **Benchmark Parameters**: `MAX_WORKERS`, `NUM_QUESTIONS`
- **Timeout Settings**: `REQUEST_TIMEOUT`, `MAX_RETRIES`, `RETRY_DELAY`

## Подходы к тестированию

### 1. Without Reasoning
**Папка**: `without_reasoning/`

Простой подход: модель получает контекст и вопрос, сразу возвращает ответ.

```bash
cd without_reasoning
python test_squad_without_reasoning.py
```

### 2. With Structured Output (Single-Step)
**Папка**: `with_structured_output/`

Одношаговый подход с JSON schema:
- Модель генерирует `reasoning` и `answer` в одном запросе через Structured Output

```bash
cd with_structured_output
python test_squad_with_so.py
```

### 3. With Two-Step Structured Output
**Папка**: `with_two_step_so/`

Двухшаговый подход с JSON schema:
1. **Шаг 1**: Модель генерирует `reasoning` через Structured Output
2. **Шаг 2**: `reasoning` вставляется в `assistant` role, модель извлекает финальный ответ как free-form content

```bash
cd with_two_step_so
python test_squad_two_step_so.py
```

### 4. With ReAct (Function Calling)
**Папка**: `with_react/`

Агентный подход с использованием Function Calling:
1. **Шаг 1**: Модель вызывает tool `generate_reasoning` для анализа
2. **Шаг 2**: После получения результата tool, модель извлекает финальный ответ

```bash
cd with_react
python test_squad_with_reasoning.py
```

## Результаты

Каждый тест сохраняет результаты в своей папке `results/`:
- **JSONL файлы**: Инкрементальное сохранение каждого ответа (для анализа на лету)
- **Summary файлы**: Итоговая статистика по завершению теста

### Метрики

- **Accuracy (EMIN)**: Exact Match In - проверка, содержится ли хотя бы один из правильных ответов в ответе модели
- **Response Time**: Среднее время ответа на вопрос
- **Throughput**: Количество вопросов в секунду
- **Questions/Second**: Скорость обработки вопросов

## 📊 Результаты бенчмарков

**Датасет**: SQuAD v1.1 Validation (10,570 вопросов)  
**Модель**: qwen3-30b-a3b-instruct-2507  
**Метрика**: EMIN (Exact Match In) - проверка содержания правильного ответа в ответе модели  
**Дата**: 28 декабря 2025

### Сводная таблица

| # | Подход | Accuracy | Correct | Incorrect | Speed (q/s) | Avg Response | Total Time | Статус |
|---|--------|----------|---------|-----------|-------------|--------------|------------|--------|
| 1 | **Without Reasoning** | **89.37%** | 9,446 | 1,124 | **15.00** | **2.00s** | **11.7 min** | ✅ |
| 2 | **Single-Step SO** | **90.98%** | 9,617 | 953 | 4.62 | 6.42s | 38.1 min | ✅ |
| 3 | **Two-Step SO** | **🏆 93.47%** | **9,880** | **690** | 2.64 | 11.35s | 66.7 min | ✅ |
| 4 | **ReAct (1 Tool)** | **91.98%** | 9,722 | 848 | 3.45 | 8.68s | 51.0 min | ✅ |
| 5 | **ReAct (2 Tools)** | **🥈 93.27%** | **9,859** | **711** | 3.31 | 9.04s | 53.2 min | ✅ |

### Детальное сравнение завершенных тестов

#### 1. Without Reasoning (Baseline)
- **Описание**: Прямой ответ без рассуждений
- **Accuracy**: 89.37%
- **Скорость**: 15.00 q/s (самый быстрый)
- **Время ответа**: 2.00s (самое быстрое)
- **Общее время**: 11.7 минут
- **Плюсы**: ⚡ Максимальная скорость, низкая латентность
- **Минусы**: ⚠️ Наименьшая точность

#### 2. Single-Step SO
- **Описание**: Reasoning + answer в одном JSON через Structured Output
- **Accuracy**: 90.98%
- **Скорость**: 4.62 q/s
- **Время ответа**: 6.42s
- **Общее время**: 38.1 минут
- **Плюсы**: 📊 Средняя точность (+1.61% vs baseline), средняя скорость
- **Минусы**: ⚠️ Уступает Two-Step SO и ReAct

#### 3. Two-Step SO 🏆
- **Описание**: Reasoning через SO, затем reasoning в assistant role + free-form answer
- **Accuracy**: 93.47% (лучший результат!)
- **Скорость**: 2.64 q/s (самый медленный)
- **Время ответа**: 11.35s (самое долгое)
- **Общее время**: 66.7 минут
- **Плюсы**: 🏆 Максимальная точность (+4.10% vs baseline), разделение reasoning и answer
- **Минусы**: 🐌 Самый медленный (в 5.7 раз медленнее baseline)

#### 4. ReAct (1 Tool - Function Calling)
- **Описание**: Модель вызывает `generate_reasoning`, затем отвечает free-form
- **Accuracy**: 91.98%
- **Скорость**: 3.45 q/s
- **Время ответа**: 8.68s
- **Общее время**: 51.0 минут
- **Плюсы**: 🤖 Агентный подход, хорошая точность
- **Минусы**: ⚠️ Уступает подходам с разделением reasoning/answer

#### 5. ReAct (2 Tools) 🥈
- **Описание**: Модель вызывает `generate_reasoning`, затем `submit_answer`
- **Accuracy**: 93.27% (второй лучший результат!)
- **Скорость**: 3.31 q/s
- **Время ответа**: 9.04s
- **Общее время**: 53.2 минут
- **Плюсы**: 🏅 Очень высокая точность (+3.90% vs baseline), структурированный answer
- **Минусы**: 🐌 Медленнее baseline в 4.5 раз

### Ключевые выводы

#### 🎯 Главные открытия

**1. Разделение reasoning и answer критично!**

Топ-2 результата используют **разделение на два шага**:
- 🏆 **Two-Step SO (93.47%)**: reasoning через SO → free-form answer
- 🥈 **ReAct 2 Tools (93.27%)**: reasoning через tool → answer через tool

Оба обогнали подходы с одновременной генерацией или одним tool call.

**2. Способ получения answer имеет значение!**

Сравнение при одинаковом reasoning подходе:
- **ReAct 1 Tool** (free-form answer): 91.98%
- **ReAct 2 Tools** (tool call answer): 93.27% (+1.29%)

Структурированный answer через tool дает прирост!

#### Точность vs Скорость

**Прирост точности от reasoning:**
- **Single-Step SO**: +1.61% (171 дополнительных правильных ответов)
- **ReAct (1 Tool)**: +2.61% (276 дополнительных правильных ответов)
- **ReAct (2 Tools)**: +3.90% (413 дополнительных правильных ответов) 🥈
- **Two-Step SO**: +4.10% (434 дополнительных правильных ответов) 🏆

**Цена за точность:**
- **Single-Step SO**: в 3.2 раза медленнее baseline (15.0 → 4.62 q/s)
- **ReAct (1 Tool)**: в 4.3 раза медленнее baseline (15.0 → 3.45 q/s)
- **ReAct (2 Tools)**: в 4.5 раз медленнее baseline (15.0 → 3.31 q/s)
- **Two-Step SO**: в 5.7 раз медленнее baseline (15.0 → 2.64 q/s)

**Латентность:**
- Baseline: 2.00s
- Single-Step SO: 6.42s (3.2x)
- ReAct (1 Tool): 8.68s (4.3x)
- ReAct (2 Tools): 9.04s (4.5x)
- Two-Step SO: 11.35s (5.7x)

#### 💡 Ключевые инсайты

**1. Reasoning работает!**
- Все подходы с reasoning показали лучшую точность, чем baseline
- Прирост от +1.61% до +4.10%
- 434 дополнительных правильных ответа из 10,570

**2. Способ интеграции reasoning имеет значение**
```
Single-Step SO (90.98%) < ReAct (91.98%) < Two-Step SO (93.47%)
```

**Почему Two-Step SO лучше:**
- ✅ Reasoning генерируется структурированно (через SO)
- ✅ Answer извлекается свободным текстом (без ограничений SO)
- ✅ Модель "видит" свои рассуждения в контексте перед ответом
- ✅ Нет конфликта между "думать" и "форматировать" одновременно

**Почему Single-Step SO хуже:**
- ❌ Модель должна одновременно рассуждать И форматировать ответ в JSON
- ❌ SO может ограничивать естественность извлечения ответа
- ❌ Конфликт задач: "анализировать" vs "структурировать"

**Почему ReAct (1 Tool) на 4-м месте:**
- ✅ Явное разделение reasoning (tool call) и answer
- ⚠️ Free-form answer менее структурирован
- ⚠️ Модель может "забыть" детали reasoning при финальном ответе

**Почему ReAct (2 Tools) на 2-м месте:**
- ✅ Явное разделение reasoning и answer через tool calls
- ✅ Структурированный answer через `submit_answer` tool
- ✅ Модель "видит" reasoning перед вызовом второго tool
- ⚠️ Немного уступает Two-Step SO (возможно, из-за overhead tool calls)

**3. Trade-off: точность vs скорость**
- За каждый +1% точности платим ~1.4x замедлением
- Two-Step SO: +4.10% точности за 5.7x замедление
- Для production нужен баланс в зависимости от задачи

**4. Архитектурные выводы**
- 🔑 **Разделяйте "думать" и "отвечать"** - это дает лучшие результаты (+1.29% - +4.10%)
- 🔑 **Два лучших паттерна:**
  - SO для reasoning → free-form answer (93.47%)
  - Tool для reasoning → tool для answer (93.27%)
- 🔑 **Структурированный answer лучше free-form** - сравните ReAct 1 vs 2 Tools (+1.29%)
- 🔑 **Контекст с reasoning критичен** - модель должна "видеть" свои рассуждения
- 🔑 **Простые промпты работают** - не нужны сложные инструкции

**5. Неожиданные результаты**
- ❗ **Two-Step SO > ReAct 2 Tools** (93.47% vs 93.27%) - free-form answer чуть лучше tool call
- ❗ **ReAct 2 Tools > ReAct 1 Tool** (93.27% vs 91.98%) - структурированный answer дает +1.29%
- ❗ **Single-Step SO худший среди reasoning** (90.98%) - одновременная генерация мешает

---

## 📊 Ответы на исследовательские вопросы

### О reasoning:

**❓ Улучшает ли reasoning точность? На сколько процентов?**
✅ **ДА!** От +1.61% до +4.10%
- Single-Step SO: +1.61% (171 доп. правильных ответа)
- ReAct 1 Tool: +2.61% (276 доп. правильных ответа)
- ReAct 2 Tools: +3.90% (413 доп. правильных ответа)
- Two-Step SO: +4.10% (434 доп. правильных ответа) 🏆

**❓ Какой способ reasoning лучше? SO vs Function Calling?**
✅ **Оба работают хорошо, но по-разному:**
- SO (Two-Step): 93.47% - лучший результат
- Function Calling (2 Tools): 93.27% - очень близко
- Разница: 0.20% (21 вопрос из 10,570)

**Вывод:** Способ получения reasoning не критичен. Важно **разделение на два шага**.

### О способе получения ответа:

**❓ Free-form vs Structured Output для answer? Что точнее?**
✅ **Free-form НЕМНОГО лучше:**
- Two-Step SO (free-form answer): 93.47%
- ReAct 2 Tools (tool call answer): 93.27%
- Разница: 0.20%

**Почему free-form лучше?**
- Модель не ограничена форматом JSON
- Может естественно извлекать фразы из контекста
- Нет overhead на структурирование

**❓ Одновременная генерация vs разделение? Мешает ли SO при генерации answer?**
✅ **ДА, мешает!**
- Single-Step SO (одновременно): 90.98%
- Two-Step SO (разделение): 93.47%
- Разница: +2.49% от разделения!

**Почему одновременная генерация хуже?**
- Конфликт задач: "думать" vs "форматировать"
- Модель пытается делать два дела сразу
- JSON артефакты в ответах ("} 4", "} 7 {")

**❓ Tool call для answer vs free-form? Дает ли явная структура преимущество?**
✅ **Да, но НЕБОЛЬШОЕ:**
- ReAct 2 Tools (tool call): 93.27%
- ReAct 1 Tool (free-form): 91.98%
- Разница: +1.29%

**НО:** Free-form после SO reasoning еще лучше (93.47%)

### О trade-offs:

**❓ Какой ценой? Как меняется скорость и латентность?**
✅ **Чем выше точность, тем медленнее:**

| Подход | Точность | Замедление | Латентность |
|--------|----------|------------|-------------|
| Without Reasoning | 89.37% | 1x | 2.00s |
| Single-Step SO | 90.98% | 3.2x | 6.42s |
| ReAct 1 Tool | 91.98% | 4.3x | 8.68s |
| ReAct 2 Tools | 93.27% | 4.5x | 9.04s |
| Two-Step SO | 93.47% | 5.7x | 11.35s |

**Формула:** За каждый +1% точности платим ~1.4x замедлением

**❓ Стоит ли оно того? Оправдывает ли прирост точности потерю скорости?**
✅ **ЗАВИСИТ ОТ ЗАДАЧИ:**

**Стоит использовать reasoning когда:**
- ✅ Критична точность (медицина, юриспруденция, финансы)
- ✅ Стоимость ошибки высока
- ✅ Низкая/средняя нагрузка
- ✅ Нужна прослеживаемость решений

**НЕ стоит использовать reasoning когда:**
- ❌ Нужна максимальная скорость (FAQ, чат-боты)
- ❌ Высокая нагрузка (тысячи запросов/сек)
- ❌ Простые вопросы с явными ответами
- ❌ Допустима точность ~89%

---

## ⚠️ Критичный инсайт для агентных фреймворков

### Проблема: Tool call для финального ответа ограничивает гибкость

**Наблюдение из эксперимента:**
```
ReAct 2 Tools (answer через tool): 93.27%
Two-Step SO (answer free-form):    93.47%
Разница: -0.20% при использовании tool
```

### Почему это важно для агентов?

**1. Агентные фреймворки часто требуют структурированный output**
```python
# Типичный паттерн в LangChain, AutoGPT, etc.
class Agent:
    tools = [
        {"name": "search", "output": SearchResult},
        {"name": "calculate", "output": CalculationResult},
        {"name": "final_answer", "output": FinalAnswer}  # ← Структурированный!
    ]
```

**Проблема:** Модель должна "впихнуть" ответ в структуру, даже если это неестественно.

**2. Примеры из нашего эксперимента:**

**Вопрос:** "What is the position Derek Wolfe plays currently?"
- **Expected:** "Defensive ends"
- **ReAct 2 Tools (tool):** "defensive end" ❌ (единственное число)
- **Two-Step SO (free-form):** "Defensive ends" ✅

**Вопрос:** "Which hotel did the Broncos use?"
- **Expected:** "Santa Clara Marriott."
- **ReAct 2 Tools (tool):** "Santa Clara Marriott" ❌ (без точки)
- **Two-Step SO (free-form):** "Santa Clara Marriott." ✅

### Рекомендации для агентных систем:

**❌ НЕ ДЕЛАЙТЕ ТАК:**
```python
# Плохо: финальный ответ через tool с жесткой структурой
tools = [
    {"name": "think", "parameters": {"reasoning": "string"}},
    {"name": "answer", "parameters": {"answer": "string"}}  # ← Ограничивает!
]
```

**✅ ДЕЛАЙТЕ ТАК:**
```python
# Хорошо: reasoning через tool, answer free-form
# Шаг 1: Структурированное reasoning
reasoning = agent.call_tool("analyze_context", {"reasoning": "..."})

# Шаг 2: Свободный ответ с контекстом
messages.append({"role": "assistant", "content": reasoning})
messages.append({"role": "user", "content": "Now extract the exact answer."})
final_answer = agent.generate(messages)  # ← Free-form!
```

**🎯 ИЛИ ГИБРИДНЫЙ ПОДХОД:**
```python
# Используйте tool для reasoning, но не для финального ответа
class SmartAgent:
    def process(self, task):
        # 1. Tool для анализа (структурировано)
        analysis = self.call_tool("analyze", task)
        
        # 2. Tool для промежуточных действий (структурировано)
        actions = self.call_tool("plan_actions", analysis)
        
        # 3. Финальный ответ БЕЗ tool (свободно)
        return self.generate_final_answer(task, analysis, actions)
```

### Выводы для разработчиков агентов:

1. **Не заставляйте модель структурировать финальный output**
   - Reasoning → структурируйте (SO или tool)
   - Final answer → оставьте свободным

2. **Tool calls хороши для промежуточных шагов**
   - Поиск информации
   - Вычисления
   - Анализ
   - НО НЕ для финального ответа пользователю

3. **Разница 0.20% кажется малой, но:**
   - На 10,000 запросов = 20 дополнительных ошибок
   - В критичных системах это важно
   - Пользовательский опыт страдает

4. **Баланс структуры и гибкости:**
   ```
   Высокая структура (tool) → Хорошо для pipeline
   Низкая структура (free-form) → Хорошо для качества ответа
   ```

### Практический пример:

```python
# ❌ Плохая архитектура агента
class RigidAgent:
    def answer_question(self, question):
        # Все через tools - слишком жестко
        reasoning = self.call_tool("think", {"question": question})
        answer = self.call_tool("answer", {"answer": "..."})  # ← Ограничено!
        return answer

# ✅ Хорошая архитектура агента  
class FlexibleAgent:
    def answer_question(self, question):
        # Reasoning через tool (структурировано)
        reasoning = self.call_tool("think", {"question": question})
        
        # Answer свободно (гибко)
        context = self.build_context(question, reasoning)
        answer = self.llm.generate(context)  # ← Свободно!
        
        return {
            "answer": answer,
            "reasoning": reasoning,  # Для прослеживаемости
            "confidence": self.estimate_confidence(answer)
        }
```

#### Рекомендации по выбору подхода

**Используйте Without Reasoning когда:**
- ⚡ Нужна максимальная скорость обработки
- 🚀 Критична низкая латентность
- ✅ Допустима точность ~89%
- 📈 Высокая нагрузка (много запросов)

**Используйте Single-Step SO когда:**
- 📊 Нужен баланс точности и скорости
- ⚡ Умеренная нагрузка
- 💰 Важна экономия на API calls (1 запрос vs 2)
- ✅ Допустима точность ~91%
- 🏃 Нужен быстрый reasoning (быстрее ReAct и Two-Step)

**Используйте ReAct когда:**
- 🤖 Нужен агентный подход (модель сама решает использовать reasoning)
- 🔧 Есть инфраструктура для function calling
- ⚖️ Нужен баланс точности (92%) и скорости (быстрее Two-Step)
- 📊 Средняя нагрузка

**Используйте Two-Step SO когда:**
- 🏆 Критична максимальная точность (93.5%)
- 💎 Качество важнее скорости
- 🎯 Сложные задачи извлечения информации
- 📉 Низкая нагрузка
- ⏱️ Допустима высокая латентность (11s)

### Файлы с результатами

**Without Reasoning:**
- Детальные: `without_reasoning/results/squad_test_20251228_123452.jsonl`
- Сводка: `without_reasoning/results/summary_20251228_123452.txt`

**Single-Step SO:**
- Детальные: `with_structured_output/results/squad_so_20251228_124756.jsonl`
- Сводка: `with_structured_output/results/summary_so_20251228_132602.txt`

**Two-Step SO:**
- Детальные: `with_two_step_so/results/squad_two_step_so_20251228_135901.jsonl`
- Сводка: `with_two_step_so/results/summary_two_step_so_20251228_162030.txt`

**ReAct (Function Calling):**
- Детальные: `with_react/results/squad_reasoning_20251228_113451.jsonl`
- Сводка: `with_react/results/summary_reasoning_20251228_122552.txt`

---

## 🚀 Применение результатов в бизнес-кейсах

### 1. Когда использовать каждый подход

#### 💼 Customer Support / FAQ Bot
**Рекомендация:** Without Reasoning или Single-Step SO

**Почему:**
- ⚡ Нужна низкая латентность (2-6s)
- 💰 Высокая нагрузка → экономия на API calls
- ✅ Допустима точность 89-91%
- 📊 Простые вопросы с явными ответами

**Пример:**
```python
# Простой подход для FAQ
response = client.chat.completions.create(
    model="qwen3-30b",
    messages=[
        {"role": "system", "content": "Answer based on FAQ context."},
        {"role": "user", "content": f"Context: {faq}\n\nQuestion: {question}"}
    ]
)
```

#### 🏥 Medical/Legal AI Assistant
**Рекомендация:** Two-Step SO или ReAct 2 Tools

**Почему:**
- 🎯 Критична максимальная точность (93%+)
- 💎 Качество важнее скорости
- 📋 Нужна прослеживаемость reasoning
- ⚖️ Ответственность за ошибки высока

**Пример:**
```python
# Шаг 1: Генерация reasoning
reasoning_response = client.chat.completions.create(
    model="qwen3-30b",
    messages=[...],
    response_format={"type": "json_schema", "schema": reasoning_schema}
)

# Шаг 2: Извлечение ответа с контекстом reasoning
messages.append({"role": "assistant", "content": reasoning})
final_response = client.chat.completions.create(
    model="qwen3-30b",
    messages=messages
)
```

#### 🤖 Autonomous Agents / Complex Workflows
**Рекомендация:** ReAct 2 Tools

**Почему:**
- 🔧 Нужна интеграция с другими tools
- 🧠 Агент должен "думать" перед действием
- 📊 Структурированный output для pipeline
- 🔄 Reasoning можно логировать/анализировать

**Пример:**
```python
tools = [
    {"name": "analyze_context", "parameters": {"reasoning": "..."}},
    {"name": "extract_answer", "parameters": {"answer": "..."}},
    {"name": "search_database", "parameters": {"query": "..."}},
]

# Агент сам решает последовательность
response = client.chat.completions.create(
    model="qwen3-30b",
    messages=messages,
    tools=tools
)
```

#### 📊 Data Extraction / Document Processing
**Рекомендация:** ReAct 2 Tools или Two-Step SO

**Почему:**
- 🎯 Нужна высокая точность извлечения
- 📋 Структурированный output критичен
- 🔍 Reasoning помогает в сложных документах
- ⚖️ Баланс точности и throughput

### 2. Паттерны проектирования агентов

#### Паттерн 1: "Think Then Act"
```python
# Основан на Two-Step SO (93.47% accuracy)
class ThinkThenActAgent:
    async def process(self, task):
        # 1. Генерируем reasoning через SO
        thinking = await self.generate_reasoning(task)
        
        # 2. Действуем на основе reasoning (free-form)
        action = await self.decide_action(task, thinking)
        
        return action, thinking  # Возвращаем оба для прослеживаемости
```

**Применение:**
- Финансовый анализ
- Юридический review
- Медицинская диагностика

#### Паттерн 2: "Tool-Based Reasoning"
```python
# Основан на ReAct 2 Tools (93.27% accuracy)
class ToolBasedAgent:
    tools = [
        {"name": "analyze", "description": "Analyze the situation"},
        {"name": "decide", "description": "Make a decision"},
        {"name": "execute", "description": "Execute action"}
    ]
    
    async def process(self, task):
        # Агент сам выбирает последовательность tools
        while not done:
            tool_call = await self.llm.call_tool(task, self.tools)
            result = await self.execute_tool(tool_call)
            task.add_context(result)
        
        return task.final_result
```

**Применение:**
- Автономные агенты
- Multi-step workflows
- Complex decision making

#### Паттерн 3: "Fast Path with Reasoning Fallback"
```python
# Комбинация подходов для оптимизации
class HybridAgent:
    async def process(self, task):
        # Пробуем быстрый путь (Without Reasoning)
        quick_answer = await self.quick_answer(task)
        confidence = self.estimate_confidence(quick_answer)
        
        if confidence > 0.95:
            return quick_answer  # 15 q/s
        
        # Fallback на reasoning для сложных случаев
        return await self.reasoning_answer(task)  # 3-4 q/s
```

**Применение:**
- Высоконагруженные системы
- Оптимизация costs
- Adaptive AI systems

### 3. Борьба с типичными ошибками

#### Проблема: "Too Long (Explanation)" - 44.8% ошибок без reasoning

**Решение 1:** Используйте reasoning подходы
```python
# ❌ Плохо: модель дает объяснения
messages = [{"role": "user", "content": "Question: ..."}]

# ✅ Хорошо: разделяем thinking и answer
# Шаг 1: думаем
# Шаг 2: отвечаем кратко
```

**Решение 2:** Добавьте constraint в prompt
```python
system_prompt = """Extract ONLY the exact answer from context. 
No explanations. Maximum 5 words."""
```

#### Проблема: "Partial Match" - 45-51% ошибок с reasoning

**Решение:** Используйте post-processing
```python
def normalize_answer(answer: str, gold_answers: List[str]) -> str:
    """Normalize format to match expected answer."""
    # Убираем артикли
    answer = re.sub(r'\b(the|a|an)\b', '', answer, flags=re.IGNORECASE)
    
    # Нормализуем числа: "5" → "five" если ожидается слово
    if any(word.isalpha() for word in gold_answers[0].split()):
        answer = number_to_word(answer)
    
    # Убираем лишнюю пунктуацию
    answer = answer.strip('.,!?')
    
    return answer
```

#### Проблема: JSON артефакты в Single-Step SO ("} 4", "} 7 {")

**Решение:** Используйте Two-Step или ReAct 2 Tools
```python
# ❌ Плохо: reasoning + answer в одном JSON
response_format = {
    "reasoning": "...",
    "answer": "..."  # ← может содержать артефакты
}

# ✅ Хорошо: разделяем на два шага
# Шаг 1: JSON для reasoning
# Шаг 2: Free-form или отдельный tool для answer
```

### 4. Метрики для мониторинга

```python
class AgentMetrics:
    """Метрики для production мониторинга."""
    
    def track_performance(self, result):
        # 1. Accuracy метрики
        self.accuracy = self.calculate_emin(result)
        
        # 2. Latency метрики
        self.p50_latency = percentile(latencies, 50)
        self.p95_latency = percentile(latencies, 95)
        self.p99_latency = percentile(latencies, 99)
        
        # 3. Error категории (из нашего анализа)
        self.error_categories = {
            "partial_match": count,
            "wrong_answer": count,
            "too_long": count,
            "format_error": count
        }
        
        # 4. Cost метрики
        self.cost_per_request = (
            prompt_tokens * INPUT_PRICE + 
            completion_tokens * OUTPUT_PRICE
        )
        
        # 5. Reasoning quality (если используется)
        if result.reasoning:
            self.reasoning_length = len(result.reasoning)
            self.reasoning_relevance = self.score_relevance(result.reasoning)
```

### 5. A/B тестирование подходов

```python
class ABTestFramework:
    """Framework для A/B тестирования разных подходов."""
    
    approaches = {
        "baseline": WithoutReasoningAgent(),
        "single_step": SingleStepSOAgent(),
        "two_step": TwoStepSOAgent(),
        "react_1": ReActOneToolAgent(),
        "react_2": ReActTwoToolsAgent(),
    }
    
    async def route_request(self, user_id: str, task):
        # Распределяем пользователей по группам
        approach = self.get_approach_for_user(user_id)
        
        # Логируем для анализа
        start = time.time()
        result = await approach.process(task)
        latency = time.time() - start
        
        self.log_metrics(
            approach=approach.name,
            user_id=user_id,
            accuracy=self.check_accuracy(result),
            latency=latency,
            cost=self.calculate_cost(result)
        )
        
        return result
```

### 6. Рекомендации по стоимости

| Подход | Requests/min | Cost/1K req | Use Case |
|--------|--------------|-------------|----------|
| Without Reasoning | 900 (15 q/s) | $X | High-volume FAQ |
| Single-Step SO | 277 (4.6 q/s) | $1.5X | Moderate accuracy |
| Two-Step SO | 158 (2.6 q/s) | $2.5X | High accuracy critical |
| ReAct 1 Tool | 207 (3.4 q/s) | $2X | Balanced approach |
| ReAct 2 Tools | 199 (3.3 q/s) | $2.2X | Agent workflows |

**Оптимизация costs:**
```python
# Используйте кэширование для reasoning
@cache(ttl=3600)
def get_reasoning(context: str, question: str):
    return generate_reasoning(context, question)

# Batch processing для снижения overhead
async def process_batch(questions: List[str]):
    return await asyncio.gather(*[
        process_question(q) for q in questions
    ])
```

---

## Установка зависимостей

```bash
pip install aiohttp rich datasets
```

## Скачивание данных

```bash
python download_squad.py
```

