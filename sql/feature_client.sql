SELECT 
	cdc.ApplicationId AS ApplicationId,
    cdc.ApplicationDate AS ApplicationDate,
	MainDebt,LoanSerialNumber,
	{target_column}
    PositionType,
    EducationType,
    OccupationType,
    MaritalStatus,
    Sex,
    OS,
    BirthDate ,
    FieldOfActivity,
    WorkExperience,
    toYear(ApplicationDate) - toYear(BirthDate) AS age,
    IsRegistrationAddressCoincides,
    IncomeAmount,
    case when IsRegistrationAddressCoincides = 1 then TimezoneA else TimezoneAA end as Timezone,
    case when ActualIsPrivateHouse is not null then ActualIsPrivateHouse else IsPrivateHouse end as ActualIsPrivateHouse ,
    CardBrand,
    CardType,
    CardLevel,
    case when IsPC =1 then 'ПК'
        when IsTablet = 1 then 'Планшет'
        when IsMobile = 1 then 'Смартфон' end as gadget
FROM risk_ch_db.client_data_495credit cdc 
{target_join}
WHERE {filters}
