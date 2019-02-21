SELECT count(*)
FROM trips
WHERE 1 = 1
AND status = 'completed' 
AND completed_at >= '2016-01-01 00:00:00' 
AND completed_at <= '2016-12-31 23:59:59'
AND driver_uuid NOT IN (SELECT uuid FROM driver WHERE is_test_account)
