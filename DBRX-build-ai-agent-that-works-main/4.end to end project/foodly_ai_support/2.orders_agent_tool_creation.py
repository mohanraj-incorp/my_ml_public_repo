# Databricks notebook source
# MAGIC %md
# MAGIC ### Creating tools for order agent , this will be uc functions

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC create schema if not exists agents.orders_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS agents.orders_data.orders (
# MAGIC     order_id STRING,
# MAGIC     user_id STRING,
# MAGIC     restaurant_name STRING,
# MAGIC     status STRING,             -- e.g. 'Placed', 'Preparing', 'Out for delivery', 'Delivered', 'Cancelled'
# MAGIC     eta STRING,                -- estimated time e.g. '20 min'
# MAGIC     rider_name STRING,
# MAGIC     delivery_address STRING,
# MAGIC     total_price DECIMAL(10,2),
# MAGIC     created_at TIMESTAMP
# MAGIC );
# MAGIC
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS agents.orders_data.order_items (
# MAGIC     order_id STRING,
# MAGIC     item_name STRING,
# MAGIC     quantity INT,
# MAGIC     price DECIMAL(10,2)
# MAGIC );
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Insert sample orders
# MAGIC INSERT INTO agents.orders_data.orders VALUES
# MAGIC ('ORD001', 'USR123', 'Pizza Palace',      'Placed',            '25 min', 'Ramesh', '221B Baker Street', 599.00, current_timestamp()),
# MAGIC ('ORD002', 'USR123', 'Burger Bazaar',     'Preparing',         '30 min', 'Suresh', '221B Baker Street', 399.00, current_timestamp() - INTERVAL 1 HOUR),
# MAGIC ('ORD003', 'USR456', 'Curry Corner',      'Out for delivery',  '15 min', 'Mahesh', '742 Evergreen Terrace', 799.00, current_timestamp() - INTERVAL 2 HOUR),
# MAGIC ('ORD004', 'USR456', 'Pasta Point',       'Delivered',         '0 min',  'Ganesh', '742 Evergreen Terrace', 499.00, current_timestamp() - INTERVAL 1 DAY);
# MAGIC
# MAGIC -- Insert sample order items
# MAGIC INSERT INTO agents.orders_data.order_items VALUES
# MAGIC ('ORD001', 'Margherita Pizza', 1, 299.00),
# MAGIC ('ORD001', 'Garlic Bread',     1, 150.00),
# MAGIC ('ORD001', 'Coke',             2, 75.00),
# MAGIC ('ORD002', 'Cheeseburger',     2, 199.50),
# MAGIC ('ORD002', 'Fries',            1, 100.00),
# MAGIC ('ORD003', 'Paneer Tikka',     1, 299.00),
# MAGIC ('ORD003', 'Butter Naan',      4, 100.00),
# MAGIC ('ORD003', 'Gulab Jamun',      2, 100.00),
# MAGIC ('ORD004', 'Alfredo Pasta',    1, 299.00),
# MAGIC ('ORD004', 'Garlic Bread',     1, 150.00),
# MAGIC ('ORD004', 'Lemonade',         1, 50.00);
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 🛒 Get all orders for a user
# MAGIC CREATE OR REPLACE FUNCTION agents.orders_data.get_all_orders(user_id_input STRING COMMENT 'User ID used to retrieve all active or recent orders')
# MAGIC RETURNS TABLE (
# MAGIC     order_id STRING,
# MAGIC     restaurant STRING,
# MAGIC     status STRING,
# MAGIC     eta STRING
# MAGIC )
# MAGIC COMMENT 'Returns all active or recent orders for the given user_id, including restaurant name, status, and ETA.'
# MAGIC RETURN (
# MAGIC     SELECT order_id, restaurant_name AS restaurant, status, eta
# MAGIC     FROM agents.orders_data.orders
# MAGIC     WHERE user_id = user_id_input
# MAGIC     ORDER BY created_at DESC
# MAGIC     LIMIT 10
# MAGIC );
# MAGIC
# MAGIC -- 📦 Get order status
# MAGIC CREATE OR REPLACE FUNCTION agents.orders_data.get_order_status(order_id_input STRING COMMENT 'Order ID to retrieve the current status')
# MAGIC RETURNS TABLE (
# MAGIC     status STRING,
# MAGIC     eta STRING,
# MAGIC     restaurant STRING,
# MAGIC     rider STRING
# MAGIC )
# MAGIC COMMENT 'Returns the current status, ETA, restaurant, and rider for the provided order_id.'
# MAGIC RETURN (
# MAGIC     SELECT status, eta, restaurant_name AS restaurant, rider_name AS rider
# MAGIC     FROM agents.orders_data.orders
# MAGIC     WHERE order_id = order_id_input
# MAGIC     LIMIT 1
# MAGIC );
# MAGIC
# MAGIC -- 🔎 Get order details
# MAGIC CREATE OR REPLACE FUNCTION agents.orders_data.get_order_details(order_id_input STRING COMMENT 'Order ID to retrieve full details')
# MAGIC RETURNS TABLE (
# MAGIC     item_name STRING,
# MAGIC     quantity INT,
# MAGIC     price DECIMAL(10,2),
# MAGIC     delivery_address STRING,
# MAGIC     placed_at TIMESTAMP
# MAGIC )
# MAGIC COMMENT 'Returns detailed information for the given order_id including items, quantity, price, delivery address, and order time.'
# MAGIC RETURN (
# MAGIC     SELECT i.item_name, i.quantity, i.price, o.delivery_address, o.created_at AS placed_at
# MAGIC     FROM agents.orders_data.order_items i
# MAGIC     JOIN agents.orders_data.orders o
# MAGIC     ON i.order_id = o.order_id
# MAGIC     WHERE o.order_id = order_id_input
# MAGIC );
# MAGIC
# MAGIC -- ❌ Cancel an order
# MAGIC CREATE OR REPLACE FUNCTION agents.orders_data.cancel_order(order_id_input STRING COMMENT 'Order ID to attempt cancellation')
# MAGIC RETURNS TABLE (
# MAGIC     success BOOLEAN,
# MAGIC     message STRING,
# MAGIC     refund_initiated BOOLEAN,
# MAGIC     refund_amount DECIMAL(10,2)
# MAGIC )
# MAGIC COMMENT 'Attempts to cancel the given order_id. Returns whether it was successful, a message, and refund details if applicable.'
# MAGIC RETURN (
# MAGIC     SELECT
# MAGIC       CASE
# MAGIC         WHEN status NOT IN ('Preparing','Out for delivery','Delivered') THEN TRUE
# MAGIC         ELSE FALSE
# MAGIC       END AS success,
# MAGIC       CASE
# MAGIC         WHEN status NOT IN ('Preparing','Out for delivery','Delivered') THEN 'Order has been cancelled successfully.'
# MAGIC         ELSE 'Order cannot be cancelled at this stage.'
# MAGIC       END AS message,
# MAGIC       CASE
# MAGIC         WHEN status NOT IN ('Preparing','Out for delivery','Delivered') THEN TRUE
# MAGIC         ELSE FALSE
# MAGIC       END AS refund_initiated,
# MAGIC       CASE
# MAGIC         WHEN status NOT IN ('Preparing','Out for delivery','Delivered') THEN total_price
# MAGIC         ELSE 0.00
# MAGIC       END AS refund_amount
# MAGIC     FROM agents.orders_data.orders
# MAGIC     WHERE order_id = order_id_input
# MAGIC     LIMIT 1
# MAGIC );
# MAGIC

# COMMAND ----------


