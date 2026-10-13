alter table admissions
      alter column deathtime type timestamp using deathtime::timestamp;

alter table icustays
    alter column intime type timestamp using intime::timestamp,
    alter column outtime type timestamp using outtime::timestamp;

alter table chartevents
    alter column charttime type timestamp using charttime::timestamp;

alter table labevents
    alter column charttime type timestamp using charttime::timestamp;