SELECT 1.0 * count(tc_filter.uuid) / count(distinct(driver_uuid)) AS average_trip_per_driver, tc_filter.city_name
FROM
(SELECT * FROM
(SELECT t.uuid, t.driver_uuid, t.request_at, c.timezone, c.city_name, c.country_name
FROM trips AS t
INNER JOIN city AS c ON t.city_uuid = c.uuid
WHERE 1 = 1
AND c.country_name = 'United State') AS tc
WHERE 1 = 1
AND CONVERT(timestamp, SWITCHOFFSET(tc.request_at, DATENAME(TzOffset, tc.timezone))) >= '2017-01-01 00:00:00'
AND CONVERT(timestamp, SWITCHOFFSET(tc.request_at, DATENAME(TzOffset, tc.timezone))) <= '2017-01-31 23:59:59') AS tc_filter
GROUP BY tc_filter.city_name
HAVING count(tc_filter.uuid) > 100000
