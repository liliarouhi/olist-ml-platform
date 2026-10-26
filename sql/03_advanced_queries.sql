-- 03_advanced_queries.sql
-- Business questions answered with CTEs, window functions and a recursive CTE.

-- Q1. Monthly revenue, month-over-month growth and running total (CTE + LAG + running SUM)
WITH monthly AS (
    SELECT date_trunc('month', o.purchased_at)::date AS month,
           sum(oi.price + oi.freight_value)          AS revenue
    FROM core.orders o
    JOIN core.order_items oi USING (order_id)
    WHERE o.order_status = 'delivered'
    GROUP BY 1
)
SELECT month,
       revenue,
       round(100 * (revenue / lag(revenue) OVER (ORDER BY month) - 1), 1) AS mom_growth_pct,
       sum(revenue) OVER (ORDER BY month)                                   AS running_total
FROM monthly
ORDER BY month;

-- Q2. Top 3 sellers by revenue in each state (window ranking with PARTITION BY)
WITH seller_revenue AS (
    SELECT s.state, s.seller_id, sum(oi.price) AS revenue
    FROM core.order_items oi
    JOIN core.sellers s USING (seller_id)
    GROUP BY 1, 2
),
ranked AS (
    SELECT *, rank() OVER (PARTITION BY state ORDER BY revenue DESC) AS rnk
    FROM seller_revenue
)
SELECT state, seller_id, revenue, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY state, rnk;

-- Q3. Repeat customers and average days between consecutive orders (LAG per customer)
WITH customer_orders AS (
    SELECT ca.customer_unique_id,
           o.purchased_at,
           lag(o.purchased_at) OVER (PARTITION BY ca.customer_unique_id
                                     ORDER BY o.purchased_at) AS prev_purchase
    FROM core.orders o
    JOIN core.customer_accounts ca USING (customer_id)
)
SELECT count(DISTINCT customer_unique_id)                                        AS customers,
       count(DISTINCT customer_unique_id) FILTER (WHERE prev_purchase IS NOT NULL) AS repeat_customers,
       round(avg(extract(epoch FROM purchased_at - prev_purchase) / 86400)::numeric, 1) AS avg_days_between_orders
FROM customer_orders;

-- Q4. Late deliveries and median delivery time per customer state (ordered-set aggregate)
SELECT ca.state,
       count(*) AS delivered_orders,
       round(100.0 * avg((o.delivered_customer_at::date > o.estimated_delivery_date)::int), 1) AS late_pct,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY o.delivered_customer_at::date - o.purchased_at::date)
           AS median_delivery_days
FROM core.orders o
JOIN core.customer_accounts ca USING (customer_id)
WHERE o.order_status = 'delivered' AND o.delivered_customer_at IS NOT NULL
GROUP BY 1
ORDER BY late_pct DESC;

-- Q5. Days without any order between the first and last purchase (recursive CTE calendar)
WITH RECURSIVE calendar(day) AS (
    SELECT min(purchased_at)::date FROM core.orders
    UNION ALL
    SELECT day + 1 FROM calendar
    WHERE day < (SELECT max(purchased_at)::date FROM core.orders)
)
SELECT c.day
FROM calendar c
LEFT JOIN core.orders o ON o.purchased_at::date = c.day
GROUP BY c.day
HAVING count(o.order_id) = 0
ORDER BY c.day;