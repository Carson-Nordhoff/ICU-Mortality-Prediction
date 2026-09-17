create table clean_mimic_data as
with admission_features as (
    select
        a.hadm_id, --id
        a.subject_id, --id

        a.hospital_expire_flag, --flag
        a.marital_status,
        a.religion,
        a.language,
        a.insurance,
        a.admission_type
    from admissions a
),
patient_features as (
    select
        p.subject_id, --id

        p.gender
    from patients p
)
select
    af.*,
    pf.gender
from admission_features af
join patient_features pf
    on af.subject_id = pf.subject_id