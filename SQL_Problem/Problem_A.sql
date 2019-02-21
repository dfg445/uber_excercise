SELECT (SELECT count(*) FROM driver WHERE NOT is_test_account) * 100 / count(*) AS not_test_driver_percentage
FROM driver
