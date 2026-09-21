--Maintain a ~24hr time cutoff for patients to prevent temporal leakage
--Add several features/tables

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
        a.admission_type,

        a.admittime --calc age
    from admissions a
),
patient_features as (
    select
        p.subject_id, --id

        p.gender,

        p.dob --calc age
    from patients p
),
icustays_features as (
    select
        icu.hadm_id, --id

        icu.first_careunit
    from icustays icu
)
select
    af.hospital_expire_flag, --flag
    af.subject_id, --id

    af.marital_status,
    af.religion,
    af.language,
    af.insurance,
    af.admission_type,

    pf.gender,

    isf.first_careunit,

    --CALCULATED INFO--
    extract(year from age(af.admittime::date, pf.dob::date)) as age --TO-DO patients age>89 have dob shifted?
from admission_features af
join patient_features pf
    on af.subject_id = pf.subject_id
join icustays_features isf
    on isf.hadm_id = af.hadm_id