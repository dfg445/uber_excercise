CREATE TABLE cancellation_rate AS 
SELECT t.driver_uuid, t.uuid AS trip_uuid,
1.0 * 
(SELECT count(*) 
FROM trips 
WHERE 1 = 1
AND t.driver_uuid = trips.driver_uuid 
AND t.request_at >= trips.request_at 
AND status = 'cancelled' ) 
/ 
(SELECT count(*) 
FROM trips 
WHERE 1 = 1
AND t.driver_uuid = trips.driver_uuid 
AND t.request_at >= trips.request_at) AS pct_cancelled,

1.0 * 
(SELECT count(*) 
FROM (SELECT *
FROM trips 
WHERE 1 = 1
AND t.driver_uuid = trips.driver_uuid 
AND t.request_at >= trips.request_at 
ORDER BY trips.request_at
LIMIT 100) AS t1
WHERE t1.status = 'cancelled') 
/ 
(SELECT count(*)
FROM (SELECT *
FROM trips 
where 1 = 1
AND t.driver_uuid = trips.driver_uuid 
AND t.request_at >= trips.request_at 
ORDER BY trips.request_at
LIMIT 100) AS t2) AS pct_cancelled_last100

FROM trips AS t
WHERE t.driver_uuid NOT IN (SELECT uuid FROM driver WHERE is_test_account)
