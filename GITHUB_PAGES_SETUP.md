# GitHub Pages Setup Instructions

## Шаги для активации GitHub Pages:

1. **Перейдите в настройки репозитория:**
   - Откройте https://github.com/vakovalskii/reasoning-benchmark-research
   - Нажмите на вкладку **Settings** (справа вверху)

2. **Найдите секцию Pages:**
   - В левом меню найдите **Pages** (в разделе "Code and automation")

3. **Настройте источник:**
   - **Source:** Deploy from a branch
   - **Branch:** `main`
   - **Folder:** `/docs`
   - Нажмите **Save**

4. **Дождитесь деплоя:**
   - GitHub автоматически задеплоит сайт
   - Процесс займет 1-2 минуты
   - Сайт будет доступен по адресу: https://vakovalskii.github.io/reasoning-benchmark-research/

5. **Проверьте результат:**
   - Откройте https://vakovalskii.github.io/reasoning-benchmark-research/
   - Вы должны увидеть красиво оформленное исследование с темой Cayman

## Структура проекта для GitHub Pages:

```
reasoning-benchmark-research/
├── docs/
│   ├── _config.yml          # Конфигурация Jekyll (тема, название)
│   └── index.md             # Главная страница (копия README.md)
├── README.md                # Главная страница GitHub репозитория
└── ...                      # Остальные файлы проекта
```

## Автоматическое обновление:

Каждый раз, когда вы обновляете `docs/index.md` и пушите в `main`, GitHub Pages автоматически обновит сайт.

Для синхронизации README с GitHub Pages:
```bash
cp README.md docs/index.md
git add docs/index.md
git commit -m "Update GitHub Pages"
git push
```

## Кастомизация темы:

Отредактируйте `docs/_config.yml` для изменения:
- Названия сайта
- Описания
- Темы (доступные темы: cayman, minimal, slate, architect, и др.)

Список тем: https://pages.github.com/themes/

