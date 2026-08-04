# Tests for Claims in v2 of the BS-DFS Paper

Test harness was generated with assistance of Claude (Anthropic), model Claude Fable 5.

## Smoke Test
```
> campaign_er.py smoke
smoke: traced, n = 4..9
  n=  4 done  outputs=427 events=4,354
  n=  5 done  outputs=1,208 events=11,732
  n=  6 done  outputs=2,891 events=25,835
  n=  7 done  outputs=7,168 events=58,756
  n=  8 done  outputs=21,066 events=158,641
  n=  9 done  outputs=86,149 events=598,623

========================================================================================================
outputs=86,149  events=598,623  capped runs=20
========================================================================================================
[ok  ] stack_nesting              -                            self       tested=104,658   viol=0
[ok  ] barrier_mirror             -                            self       tested=106,649   viol=0
[ok  ] cost_model                 sec:work-attribution         self       tested=2,380     viol=0
[ok  ] entry_guard                alg:search                   proven     tested=102,258   viol=0
[ok  ] write_values               alg:search/alg:fruitful      proven     tested=106,649   viol=0
[ok  ] write_monotonicity         lem:fruitless-increasing     proven     tested=106,649   viol=0
[ok  ] barrier_range              cor:barrier-upper            proven     tested=106,649   viol=0
[ok  ] ascent_purity              lem:phase-a-purity           proven     tested=86,832    viol=0
[ok  ] call_phase_1               cor:call-phase(1)            proven     tested=2,232     viol=0
[ok  ] call_phase_2               cor:call-phase(2)            proven     tested=7,273     viol=0
[ok  ] call_phase_3               cor:call-phase(3)            proven     tested=97,259    viol=0
[ok  ] descent_origins            lem:spine-tail-origins       proven     tested=97,259    viol=0
[ok  ] descent_seal               lem:period-seal              proven     tested=15        viol=0
[ok  ] descent_stack_1            lem:descent-stack(1)         proven     tested=59,022    viol=0
[ok  ] descent_stack_2            lem:descent-stack(2)         proven     tested=41,024    viol=0
[ok  ] raise_chain                lem:raise-bound              proven     tested=7,273     viol=0
[ok  ] drop_chain                 lem:drop-bound               proven     tested=2,117     viol=0
[ok  ] block_sequence             cor:block-sequence           proven     tested=8,189     viol=0
[ok  ] two_entries_interval       cor:two-entries              proven     tested=80        viol=0
[ok  ] two_entries_split          (new, corollary)             derived    tested=80        viol=0
[ok  ] one_entry_phase            (new)                        derived    tested=104,658   viol=0
[ok  ] worst_case_delay           thm:worst-case-delay         proven     tested=88,529    viol=0
[ok  ] initial_interval           cor:initial-interval         proven     tested=2,400     viol=0
[ok  ] amortized_delay            thm:amortized-delay          proven     tested=86,149    viol=0
[ok  ] period_budget              thm:amortized-delay(pf)      derived    tested=86,129    viol=0
[ok  ] avg_delay_2km              -                            conjecture tested=86,149    viol=0
--------------------------------------------------------------------------------------------------------
violations: 0
```

## Fast Test

```
> campaign_er.py fast 3,4,5,6,7,8,9,10 1000000
  n=  3 done  cumulative outputs=834,972
  n=  4 done  cumulative outputs=1,992,343
  n=  5 done  cumulative outputs=3,985,889
  n=  6 done  cumulative outputs=8,397,348
  n=  7 done  cumulative outputs=21,302,389
  n=  8 done  cumulative outputs=70,533,988
  n=  9 done  cumulative outputs=304,195,877
  n= 10 done  cumulative outputs=949,938,213
```

## Traced Test

```
campaign_er.py" traced 3,4,5,6,7,8,10,12,15,20 10000
  n=  3 done  outputs=8,265 events=75,730
  n=  4 done  outputs=19,687 events=188,504
  n=  5 done  outputs=39,181 events=373,354
  n=  6 done  outputs=82,530 events=740,633
  n=  7 done  outputs=208,725 events=1,690,467
  n=  8 done  outputs=720,310 events=5,215,275
  n= 10 done  outputs=3,083,568 events=22,169,021
  n= 12 done  outputs=7,302,803 events=54,842,280
  n= 15 done  outputs=13,723,714 events=109,355,119
  n= 20 done  outputs=22,574,452 events=194,526,093

========================================================================================================
outputs=22,574,452  events=194,526,093  capped runs=9,377
========================================================================================================
[ok  ] stack_nesting              -                            self       tested=34,702,097 viol=0
[ok  ] barrier_mirror             -                            self       tested=37,381,267 viol=0
[ok  ] cost_model                 sec:work-attribution         self       tested=90,623    viol=0
[ok  ] entry_guard                alg:search                   proven     tested=34,602,097 viol=0
[ok  ] write_values               alg:search/alg:fruitful      proven     tested=37,381,267 viol=0
[ok  ] write_monotonicity         lem:fruitless-increasing     proven     tested=37,381,267 viol=0
[ok  ] barrier_range              cor:barrier-upper            proven     tested=37,381,267 viol=0
[ok  ] ascent_purity              lem:phase-a-purity           proven     tested=22,592,629 viol=0
[ok  ] call_phase_1               cor:call-phase(1)            proven     tested=1,381,552 viol=0
[ok  ] call_phase_2               cor:call-phase(2)            proven     tested=3,365,578 viol=0
[ok  ] call_phase_3               cor:call-phase(3)            proven     tested=31,243,505 viol=0
[ok  ] descent_origins            lem:spine-tail-origins       proven     tested=31,243,505 viol=0
[ok  ] descent_seal               lem:period-seal              proven     tested=60,664    viol=0
[ok  ] descent_stack_1            lem:descent-stack(1)         proven     tested=18,254,583 viol=0
[ok  ] descent_stack_2            lem:descent-stack(2)         proven     tested=15,688,957 viol=0
[ok  ] raise_chain                lem:raise-bound              proven     tested=3,366,263 viol=0
[ok  ] drop_chain                 lem:drop-bound               proven     tested=2,771,499 viol=0
[ok  ] block_sequence             cor:block-sequence           proven     tested=4,460,534 viol=0
[ok  ] two_entries_interval       cor:two-entries              proven     tested=91,651    viol=0
[ok  ] two_entries_split          (new, corollary)             derived    tested=91,651    viol=0
[ok  ] one_entry_phase            (new)                        derived    tested=34,702,097 viol=0
[ok  ] worst_case_delay           thm:worst-case-delay         proven     tested=22,665,075 viol=0
[ok  ] initial_interval           cor:initial-interval         proven     tested=100,000   viol=0
[ok  ] amortized_delay            thm:amortized-delay          proven     tested=22,574,452 viol=0
[ok  ] period_budget              thm:amortized-delay(pf)      derived    tested=22,565,075 viol=0
--------------------------------------------------------------------------------------------------------
violations: 0
```
