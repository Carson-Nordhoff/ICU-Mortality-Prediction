create table clean_mimic_data as
select hospital_expire_flag,
       marital_status,
       religion,
       language,
       insurance,
       admission_type
from admissions