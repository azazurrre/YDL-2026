# YDL 2026 · Неделя 1 · День 3 · Утро
## Визуализация и история данными — Titanic, затем 911

---

# Часть 1 · Titanic — те же вопросы, теперь картинками

**0.**
```
Load the built-in "titanic" dataset from seaborn, set up matplotlib and seaborn for clean inline plots, and show head().
```

## Ответы вчерашних вопросов — визуально

**1.**
```
Show how many survived vs died as a bar chart. Give the chart a title that states the conclusion, not the topic.
```

**2.**
```
Show the survival rate of men vs women as a bar chart. Title it with the takeaway.
```

**3.**
```
Show the survival rate by passenger class as a bar chart, sorted, with a conclusion title.
```

**4.**
```
Show the distribution of passenger age as a histogram. What shape does it have?
```

**5.**
```
Show the distribution of fare as a histogram. Point out what looks unusual in the shape.
```

## Где один график говорит больше числа

**6.**
```
Plot the age distribution of survivors vs non-survivors on the same chart. Does age matter?
```

**7.**
```
Show fare by passenger class as a box plot. What does it reveal that a single average per class would hide?
```

**8.**
```
What is the average fare? Now show the full fare distribution. Does that single number describe the data well?
```

**9.**
```
Show survival by class and sex together as grouped bars. Which single group stands out?
```

**10.**
```
Make a scatter plot of age vs fare, colored by survived. Is there any visible pattern?
```

**11.**
```
Make a heatmap of survival rate with class on one axis and sex on the other.
```

## Приёмы рассказчика

**12.**
```
Take the survival-rate-by-class chart and rewrite its title so it states the conclusion in plain words.
```

**13.**
```
In the survival-by-class chart, gray out every bar except the one that matters most, and highlight that one in color.
```

**14.**
```
Show survival rate by class as small multiples: one panel per sex, side by side. What pattern appears?
```

**15.**
```
Build a 3-chart story that leads a reader to one clear conclusion about who survived the Titanic. Each chart's title should be its point.
```

---

# Часть 2 · 911 — Montgomery County

**16.**
```
Load the 911 calls dataset from 911.csv into df. Show shape, head, columns and dtypes.
```

**17.**
```
The title column looks like "EMS: BACK PAINS/INJURY". Create a new column "reason" with just the part before the colon (EMS / Fire / Traffic). Show value_counts.
```

**18.**
```
Parse the timeStamp column into a real datetime, then add columns for hour, day of week, month, and date.
```

## Вопросы к данным — картинками

**19.**
```
Which reason is most common — EMS, Fire or Traffic? Show it as a bar chart with a conclusion title.
```

**20.**
```
Show the number of calls per month over the whole period as a line. Describe the trend.
```

**21.**
```
Show the number of calls by hour of the day. When do 911 calls peak?
```

**22.**
```
Show the number of calls by day of the week, ordered Monday to Sunday.
```

**23.**
```
Show the top 10 specific call types (the full title) as a horizontal bar chart.
```

**24.**
```
Show the top 10 townships by number of calls as a horizontal bar chart.
```

## Истории, которые видно только глазами

**25.**
```
Make a heatmap with hour of day on one axis and day of week on the other, showing call volume. Where are the hot spots?
```

**26.**
```
Plot calls by hour of day as one line per reason (EMS, Fire, Traffic) on the same chart. Do they peak at different times?
```

**27.**
```
Plot every call as a point on lng (x) vs lat (y), colored by reason. What does the shape of the county look like?
```

**28.**
```
Show the monthly trend as one line per reason. Are they all growing the same way?
```

**29.**
```
What is the average number of calls per day as a single number? Now plot daily call counts over the whole period. Does the average describe it well?
```

**30.**
```
Find the single busiest day in the data. Plot that day's calls by hour, and look at what call types spiked — what likely happened?
```

## Приёмы рассказчика на 911

**31.**
```
On the calls-by-hour chart, mark and label the peak hour directly on the plot with an arrow or annotation.
```

**32.**
```
In the calls-by-day-of-week chart, highlight Traffic in color and gray out the rest. What stands out?
```

**33.**
```
Show the hourly pattern of Traffic calls vs EMS calls overlaid. Tell the rush-hour story in one chart with a conclusion title.
```

## Кульминация

**34.**
```
Many people assume Fire calls are the most common 911 emergency. Build 2-3 charts that show what is actually most common and how Fire really compares.
```

**35.**
```
Build a 4-5 chart visual story that leads a reader to one clear insight about 911 calls in this county. Each chart titled with its point, ending on the strongest one.
```

---

## Резерв

**36.**
```
Is total call volume growing month over month? Show the percent change between the first full month and the last.
```

**37.**
```
Which township has the highest SHARE of Traffic calls (not just the highest count)? Show the top 10 as a bar chart.
```

**38.**
```
Make a heatmap of townships (top 15) vs reason, showing the number of calls in each combination.
```

**39.**
```
Split the day into morning, afternoon, evening, night and show how the mix of reasons changes across them.
```

---

# Часть 3 · tips — кто хорошо даёт чаевые

**40.**
```
Load the built-in "tips" dataset from seaborn. Show head and dtypes.
```

**41.**
```
Make a scatter plot of total_bill vs tip. Does a bigger bill mean a bigger tip? Give it a conclusion title.
```

**42.**
```
Create a tip_pct column = tip / total_bill * 100. Show its distribution as a histogram.
```

**43.**
```
Show the average tip percent by day of the week as a bar chart, with a conclusion title.
```

**44.**
```
Show the average tip percent for smokers vs non-smokers. Is the difference real, or tiny? Keep the y-axis honest (start at zero).
```

**45.**
```
Show the average tip percent by party size. What is the surprising pattern?
```

**46.**
```
Box plot of tip percent for lunch vs dinner.
```

**47.**
```
In the tip-percent-by-party-size chart, color the solo-diner bar and gray out the rest. State the takeaway in the title.
```

**48.**
```
People assume bigger groups tip more generously. Build 2 charts that test this using tip PERCENT, not the total tip amount. What do they show?
```

**49.**
```
Build a 3-chart story answering "who tips well and who doesn't?", ending on the strongest finding. Each title is its point.
```

---

# Часть 4 · flights — данные во времени

**50.**
```
Load the built-in "flights" dataset from seaborn. Show head and explain what one row means.
```

**51.**
```
Plot total passengers per month over the whole period as a single line. Describe the shape.
```

**52.**
```
That line has two things happening at once. Separate them: one chart of the long-term trend (per year) and one of the within-year seasonal pattern (per month).
```

**53.**
```
Plot one line per year with month on the x-axis, all on one chart. What repeats every single year?
```

**54.**
```
Make a heatmap with years on one axis and months on the other, showing passengers. Where are the hot spots?
```

**55.**
```
What is the average passengers per month as a single number? Now plot the monthly values over time. Does that one number describe the data well?
```

**56.**
```
By how many times did air travel grow from the first year to the last? Show it as a chart with a conclusion title.
```

**57.**
```
Build a 3-chart story showing how air travel changed from 1949 to 1960, ending on the clearest chart.
```

---

# Часть 5 · World Happiness 2019 — ломаем «очевидные» мифы

**0.**
```
Load 2019.csv into df. Set up matplotlib and seaborn for clean inline plots. Show head() and the list of factor columns (GDP per capita, Social support, Healthy life expectancy, Freedom to make life choices, Generosity, Perceptions of corruption).
```

## Миф 1 · «Счастье — это в первую очередь деньги»

**1.**
```
Make a scatter plot of GDP per capita (x) vs happiness Score (y). Money clearly matters — give it a title that states that obvious starting point, not the topic.
```

**2.**
```
Compute the correlation of every factor with Score and show them as a sorted horizontal bar chart. Title it with the takeaway: is GDP really far ahead of the rest?
```

**3.**
```
Show GDP per capita, Social support and Healthy life expectancy correlations with Score side by side. They are almost equal (~0.79, 0.78, 0.78) — write a title saying money is NOT uniquely important.
```

**4.**
```
Fit a simple linear regression of Score on GDP per capita alone and report R-squared. Then add Social support and Healthy life expectancy and report the new R-squared. Show how much money-alone leaves unexplained.
```

**5.**
```
Find countries with above-average GDP but below-average Score, and countries with below-average GDP but above-average Score. Show both groups in a table. These are the counter-examples that break "money buys happiness".
```

**6.**
```
On the GDP-vs-Score scatter, highlight in color the countries that are rich-but-unhappy and poor-but-happy, gray out the rest, and label a few. Title states: money helps, but it doesn't decide.
```

**7.**
```
Build a 3-chart story that leads a reader from "happiness is mostly money" to "social support and health matter just as much". Each chart's title is its point, ending on the strongest one.
```

## Миф 2 · «Соседние страны / один регион → похожий уровень счастья»

**8.**
```
The dataset has no region column. Add one: map each country to its world region (e.g. Western Europe, Latin America, Sub-Saharan Africa, Post-Soviet/Central Asia, Middle East, South Asia, East Asia, North America, Oceania). Show value_counts of the new region column.
```

**9.**
```
Show the average happiness Score per region as a sorted bar chart with a conclusion title. This is the "regions differ" starting point.
```

**10.**
```
Now show the FULL spread of Score within each region as a box plot (or strip plot), sorted by median. Where geography looks tidy on averages, the spread tells another story — title it with that.
```

**11.**
```
For each region compute the gap between its happiest and least happy country (max Score - min Score). Show these gaps as a bar chart. Which region is the least uniform?
```

**12.**
```
Pick the region with the widest internal gap and plot its countries' Scores as a labeled bar chart. Name the happiest and the unhappiest neighbor directly on the plot.
```

**13.**
```
Compare within-region variation to between-region variation: overlay the regional means as points on top of the full distribution of all countries. Does region explain most of the differences, or only a little?
```

**14.**
```
Build a 3-chart story that breaks the claim "neighbouring countries are similarly happy": start from regional averages, reveal the within-region spread, and end on the single region where neighbours are most unequal. Each title is its point.
```
