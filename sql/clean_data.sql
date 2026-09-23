--Maintain a ~24hr time cutoff for patients to prevent temporal leakage
--Add several features/tables

alter table icustays
    alter column intime type timestamp using intime::timestamp;

alter table chartevents
    alter column charttime type timestamp using charttime::timestamp;

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
        icu.icustay_id, --id

        icu.first_careunit
    from icustays icu
),
chartevent_features as (
    with intime as (
        select
            icu.intime,
            icu.icustay_id
        from icustays icu
    )
    select
        ce.icustay_id, -- id

        avg(
            case when ce.itemid in (220045, 211) then ce.valuenum end
        ) as avg_heart_rate,

        avg(
            case when ce.itemid in (220050, 220179, 51, 442, 455, 6701) then ce.valuenum end
        ) as avg_systolic_bp,

        avg(
            case when ce.itemid in (8368, 8440, 8441, 220180, 220051) then ce.valuenum end
        ) as avg_diastolic_bp,

        avg(
            case when ce.itemid in (615, 618, 220210, 224690) then ce.valuenum end
        ) as avg_respiratory_rate,

        avg(
            case when ce.itemid in (676, 678, 223761, 223762) then ce.valuenum end
        ) as avg_body_temp,

        avg(
            case when ce.itemid in (646, 220277) then ce.valuenum end
        ) as avg_spo2

    from chartevents ce
    join intime it
        on ce.icustay_id = it.icustay_id
    where ce.itemid in (
        220045, 211, --heart rate
        220050, 220179, 51, 442, 455, 6701, --systolic_bp
        8368, 8440, 8441, 220180, 220051, --diastolic_bp
        615, 618, 220210, 224690, --respiratory_rate
        676, 678, 223761, 223762, --body_temp
        646, 220277 --spo2
        )
          and ce.charttime >= it.intime - interval '6 hours'
          and ce.charttime < it.intime + interval '1 day'
    group by ce.icustay_id
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

    cef.avg_heart_rate,
    cef.avg_systolic_bp,
    cef.avg_diastolic_bp,
    cef.avg_respiratory_rate,
    cef.avg_body_temp,
    cef.avg_spo2,

    --CALCULATED INFO--
    extract(year from age(af.admittime::date, pf.dob::date)) as age --TO-DO patients age>89 have dob shifted?
from admission_features af
join patient_features pf
    on af.subject_id = pf.subject_id
join icustays_features isf
    on isf.hadm_id = af.hadm_id
join chartevent_features cef
    on cef.icustay_id = isf.icustay_id