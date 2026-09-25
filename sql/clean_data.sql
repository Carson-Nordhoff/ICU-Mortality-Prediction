--Maintain a ~24hr time cutoff for patients to prevent temporal leakage
--Add several features/tables

alter table icustays
    alter column intime type timestamp using intime::timestamp;

alter table chartevents
    alter column charttime type timestamp using charttime::timestamp;

alter table labevents
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

        icu.intime, --for calcs

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
            case
                when ce.itemid in (676, 223762) then ce.valuenum --celcius
                when ce.itemid in (678, 223761) then (ce.valuenum-32)*(5.0/9) --convert fahrenheit to calcius
            end
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
),
labevent_features as (
    with intime as (
        select
            icu.intime,
            icu.hadm_id,
            icu.icustay_id
        from icustays icu
    )
    select
        it.icustay_id, --id

        avg(case when le.itemid = 50912 then le.valuenum end) as avg_creatinine,
        avg(case when le.itemid = 51006 then le.valuenum end) as avg_bun,
        avg(case when le.itemid = 51301 then le.valuenum end) as avg_wbc,
        avg(case when le.itemid = 51222 then le.valuenum end) as avg_hemoglobin,
        avg(case when le.itemid = 51265 then le.valuenum end) as avg_platelets,
        avg(case when le.itemid = 50983 then le.valuenum end) as avg_sodium,
        avg(case when le.itemid = 50971 then le.valuenum end) as avg_potassium,
        avg(case when le.itemid = 50931 then le.valuenum end) as avg_glucose,
        avg(case when le.itemid = 50813 then le.valuenum end) as avg_lactate

    from labevents le
    join intime it
        on le.hadm_id = it.hadm_id
    where le.itemid in (
        50912, --creatinine
        51006, --BUN (Blood Urea Nitrogen)
        51301, --WBC (White Blood Cell Count)
        51222, --hemoglobin
        51265, --platelets
        50983, --sodium
        50971, --potassium
        50931, --glucose
        50813 --lactate
        )
        and le.charttime >= it.intime - interval '6 hours'
        and le.charttime < it.intime + interval '1 day'
    group by it.icustay_id
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

    --vitals
    cef.avg_heart_rate,
    cef.avg_systolic_bp,
    cef.avg_diastolic_bp,
    cef.avg_respiratory_rate,
    cef.avg_body_temp,
    cef.avg_spo2,

    --labs
    lef.avg_creatinine,
    lef.avg_bun,
    lef.avg_wbc,
    lef.avg_hemoglobin,
    lef.avg_platelets,
    lef.avg_sodium,
    lef.avg_potassium,
    lef.avg_glucose,
    lef.avg_lactate,

    --CALCULATED INFO--
    case
        when extract(year from age(isf.intime::date, pf.dob::date)) > 89 then 91.4
        else extract(year from age(isf.intime::date, pf.dob::date))
    end as age
from admission_features af
join patient_features pf
    on af.subject_id = pf.subject_id
join icustays_features isf
    on isf.hadm_id = af.hadm_id
join chartevent_features cef
    on cef.icustay_id = isf.icustay_id
join labevent_features lef
    on lef.icustay_id = isf.icustay_id