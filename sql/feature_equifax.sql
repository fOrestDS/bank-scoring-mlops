SELECT ec.ApplicationId AS "ec.ApplicationId",
    count(cred_sum) filter (where cred_full_cost>200 and cred_active not in (0,2,5)) as active_count,
    sum(cred_sum) filter (where cred_full_cost>200 and cred_active not in (0,2,5)) as active_debt,
    count(cred_sum) filter (where cred_full_cost>200 and (cred_date + interval 30 DAY) >= ApplicationDate) as open_count_30d,
    avg(cred_sum) filter (where cred_full_cost>200 and (cred_date + interval 90 DAY) >= ApplicationDate ) as avg_check_90,
    count(cred_sum) filter(where cred_full_cost>200 and cred_active = 0 and (cred_active_date + interval 14 DAY) >= ApplicationDate) as closed_count_14,
    count(cred_sum) filter(where cred_full_cost>200 and cred_active = 0 and (cred_active_date + interval 30 DAY) >= ApplicationDate) as closed_count_30,
    count(cred_sum) filter(where cred_full_cost>200 and cred_active = 0 and (cred_active_date + interval 60 DAY) >= ApplicationDate) as closed_count_60,
    sum(cred_sum) filter(where cred_full_cost>200 and cred_active = 0 and (cred_active_date + interval 14 DAY) >= ApplicationDate) as closed_sum_14,
    sum(cred_sum) filter(where cred_full_cost>200 and cred_active = 0 and (cred_active_date + interval 30 DAY) >= ApplicationDate) as closed_sum_30,
    sum(cred_sum) filter(where cred_full_cost>200 and cred_active = 0 and (cred_active_date + interval 60 DAY) >= ApplicationDate) as closed_sum_60,
	sum(cred_day_overdue) as cred_day_overdue_sum,
	count(cred_sum) filter (where cred_active >0 and cred_day_overdue>0) as count_over_loan,
	sum(cred_day_overdue) / sum(cred_sum) filter (where cred_full_cost>200) AS OVERDUE_SHARE_SUM,
	count(cred_sum) filter (where cred_active >0 and cred_day_overdue>0 and  cred_full_cost>200) / count(cred_sum) filter (where cred_full_cost>200) as mrz_overdue_share,
	AVG(cred_sum_overdue) filter (where cred_full_cost>200 ) as avg_cred_sum_overdue,
	AVG(cred_sum_overdue) filter (where cred_full_cost>200 and (cred_date + interval 90 DAY) >= ApplicationDate ) as avg_cred_sum_overdue_90d,
	avg(delay5) filter (where cred_full_cost>200 and cred_day_overdue=0 and (cred_date + interval 90 DAY) >= ApplicationDate ) as fpd5_90D_percent,
	avg(delay30) filter (where cred_full_cost>200 and cred_day_overdue=0 ) as fpd30_percent,
	avg(ta_cred_sum_paid) filter (where cred_full_cost>200 ) as avg_cred_sum_paid,
	avg(ta_cred_sum_paid) filter (where cred_full_cost>200 and (cred_date + interval 45 DAY) >= ApplicationDate ) as avg_cred_sum_paid_45,
	avg(cred_prolong) as avg_cred_prolong,
	sum(cred_prolong) filter (where (cred_date + interval 120 DAY) >= ApplicationDate ) as count_cred_prolong_120d,
	datediff('day', min(cred_date) filter (where cred_full_cost>200), today()) as CH_length,
	avg(datediff('day',cred_date,cred_active_date)) filter (where cred_full_cost>200 and cred_active = 0) as avg_fact_term,
	max(cred_max_overdue) as max_overdue_days,
	count(cred_sum) filter (where cred_active in (2,5) and cred_full_cost>200 and (cred_active_date + INTERVAL 180 DAY )>= ApplicationDate) as count_cession_6m,
    count(cred_sum) filter (where cred_active = 0 and cred_active_date> latest.ces_date) as closed_after_one_cession
FROM risk_ch_db.equifax_credit_495credit ec
LEFT JOIN  (
    SELECT max(cred_active_date) as ces_date,ApplicationId
    FROM risk_ch_db.equifax_credit_495credit 
    WHERE  cred_active in (2,5)
    GROUP BY ApplicationId
) latest on latest.ApplicationId = ec.ApplicationId
LEFT JOIN  (
    SELECT max(cred_day_overdue) as cred_day_overdue1,ApplicationId
    FROM risk_ch_db.equifax_credit_495credit 
    WHERE  cred_active in (1,10,11,12,13,14,15,16,17,18,19)
    GROUP BY ApplicationId
) mo on mo.ApplicationId = ec.ApplicationId
WHERE {filters}
GROUP BY ec.ApplicationId
