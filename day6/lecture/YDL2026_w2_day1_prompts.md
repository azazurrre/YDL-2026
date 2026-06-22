# YDL 2026 · Неделя 2 · День 1 — Статистический анализ
## Сценарий промптов для Claude Code (утро)

Промпты вводишь ты, разбор ведёте вместе. Перед стартом дай агенту контекст
(или положи в папку CLAUDE.md из Дня 2 Недели 1 — тот же режим: дополняет ноутбук,
сам не запускает, отвечает буквально и наивно, без защитной обработки).

Первым промптом — загрузка данных, дальше всё опирается на готовый df.

---

## БЛОК 1 · 10:00–11:30 — одна переменная и её форма

### Загрузка

```
Load the seaborn 'titanic' dataset into df. Show head() and describe().
```

### Среднее против медианы — где центр врёт

```
Compute the mean and the median of the 'fare' column. Print both.
```

```
Plot a histogram of 'fare'. Draw a vertical line at the mean and another at the median, with a legend.
```

### Разброс — считаем стандартное отклонение

(сначала руками на доске: ряд {2, 10, 8, 6, 3, 7})

```
Here is a small series: [2, 10, 8, 6, 3, 7]. Compute its standard deviation with numpy and show the steps: deviations from the mean, squared, averaged, square-rooted.
```

```
Generate two samples of 1000 values each with the SAME mean (say 50) but different standard deviations (one with sd=5, one with sd=20). Plot both as overlaid histograms.
```

### Форма — нормальное распределение

```
Plot a histogram of the 'age' column from titanic. Overlay a normal curve fitted to its mean and std. Does the data look normal?
```

```
Plot a standard normal distribution curve. Shade and label the regions for 1, 2, and 3 standard deviations (the 68-95-99.7 rule).
```

### z-score — стандартизация и выбросы

```
Standardize the 'fare' column into z-scores (subtract mean, divide by std). Add it as a new column. Show the rows where the absolute z-score is greater than 3.
```

```
IQ scores have mean 100 and standard deviation 15. Compute the z-score for an IQ of 140, and what percentile that corresponds to.
```

---

## БЛОК 2 · 11:45–13:15 — от двух переменных к правде

### Связь — корреляция

```
Load the seaborn 'penguins' dataset into a new dataframe pg. Drop rows with missing values. Show head().
```

```
Make a scatter plot of bill_length_mm vs body_mass_g. Compute and print the Pearson correlation coefficient between them.
```

```
Compute the correlation matrix of all numeric columns in penguins and show it as a heatmap with the numbers annotated.
```

### Корреляция врёт — парадокс Симпсона

```
Make a scatter plot of bill_depth_mm vs body_mass_g for all penguins, and print the overall Pearson correlation.
```

```
Now make the same scatter plot of bill_depth_mm vs body_mass_g, but color the points by species, and fit a separate trend line for each species. Print the correlation within each species.
```

### От выборки к правде — смещение

```
Compute the mean fare of all titanic passengers. Then compute the mean fare of only the survivors who were in first class. Compare the two numbers.
```

### Центральная предельная теорема — симуляция (~10 мин)

```
Create a strongly skewed population of 100000 values (for example from an exponential distribution). Plot its histogram - it should look nothing like a normal curve.
```

```
From that skewed population, repeatedly draw random samples of size 30 and record each sample's mean. Do this 1000 times. Plot the histogram of the 1000 sample means.
```

```
Now repeat with sample size 5 and with sample size 100, side by side, so we can see how larger samples make the distribution of means tighter and more normal.
```

### p-value — что значит «значимо» (подводка к лабе)

```
Run a t-test comparing the fare of passengers who survived vs those who did not. Print the t-statistic and the p-value, and state plainly what the p-value means here.
```

---

## Резерв (если идёт быстро)

```
Compare the spread of 'fare' in first class vs third class using a box plot.
```

```
Show how a single extreme outlier added to a small dataset moves the mean but barely moves the median.
```

```
Take a 10% random sample of titanic and compare its survival rate to the full dataset. Repeat a few times to show how much the estimate wobbles.
```

---

## Что НЕ берём сегодня (филлер для этого дня)

Биномиальное и Пуассона, перестановки и сочетания, ANOVA, хи-квадрат,
формулы t-теста руками, сама линейная регрессия (это завтра, День 2).
Кому захочется глубже — отправляй в полный модуль по теории вероятностей и статистике.
