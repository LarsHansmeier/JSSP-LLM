# OR Lab B – Assignment 1: JSSP mit MIP & Fix-and-Optimize

## Ordnerstruktur

```
or-lab-b-a1/
├── src/
│   ├── JSSPLab.jl      Hauptmodul (bindet alles ein)
│   ├── instance.jl     Instanz-Datenstruktur + Einlesen (Standard- & Taillard-Format)
│   ├── jsplib.jl       Zugriff auf JSPLIB-Benchmarks (load_jsplib, list_jsplib, jsplib_info)
│   ├── schedule.jl     Lösungsdarstellung, Makespan, Maschinenreihenfolgen
│   ├── checker.jl      Unabhängiger Feasibility-Checker
│   ├── initial.jl      Deterministische Startlösung (Prioritätsregel)
│   ├── bounds.jl       Untere Schranken (trivial, Ein-Maschinen-Relaxierung)
│   ├── model.jl        MIP-Modell (Manne, Big-M) + Anytime-Callback
│   ├── matheuristic.jl Fix-and-Optimize (random_fao, austauschbares select_free)
│   └── reference_neighborhoods.jl  window_free (Referenz fürs Tuning, keine Policy)
├── scripts/
│   ├── setup.jl        Einmalig: Pakete installieren, Gurobi testen
│   ├── run_example.jl  Lädt Beispielinstanzen, baut Startlösung, prüft sie
│   ├── run_anytime.jl  Anytime-Performance (gurobi | random_fao | both): CSV + Grafiken
│   ├── tune_fao.jl     Parameter-Studie für random_fao (--selector=random|window)
│   └── compare_tuning.jl  Vergleich random vs. window (results/tuning_window/)
├── test/runtests.jl    Automatische Tests
├── data/instances/     Instanzdateien (example3x3, ft06)
│   └── jsplib/         162 JSPLIB-Benchmarks + metadata.csv (Optimum/Schranken)
├── results/            Ergebnisse (CSV) – nicht im Git
├── logs/               Suchverläufe – nicht im Git
├── prompts/            Dokumentation der LLM-Nutzung
└── report/             Bericht & Präsentation
```

## Erste Schritte

```bash
cd or-lab-b-a1
julia scripts/setup.jl                       # einmalig
julia --project=. test/runtests.jl           # Tests
julia --project=. scripts/run_example.jl     # Beispiel
```

In VS Code: Ordner öffnen, unten links die Julia-Umgebung auf diesen Ordner setzen
(oder in der REPL `] activate .`).

## Konventionen

- Alles 1-basiert. Operation `(j, k)` = k-te Operation von Job j.
- `inst.machine[j,k]`, `inst.ptime[j,k]`, Lösung als `Schedule(start)` mit `start[j,k]`.
- Jede Lösung (auch von Gurobi) wird mit `check_schedule` geprüft.

## Benchmark-Instanzen (JSPLIB)

Aus https://github.com/tamy0612/JSPLIB: abz5–9, ft06/10/20, la01–40, orb01–10,
swv01–20, ta01–80, yn1–4 (alle im Standardformat).

```julia
inst = load_jsplib("la01")
jsplib_info("la01")      # (name, jobs, machines, optimum, lower, upper)
list_jsplib(prefix = "la", max_jobs = 15)      # gefilterte Namensliste
```

`optimum === nothing` → nur Schranken bekannt (für ta71–80 auch keine Schranken).
Hinweis: orb07 enthält eine Operation mit Bearbeitungszeit 0.

## Anytime-Performance: Gurobi vs. Random Fix-and-Optimize

Zwei Lösungsmethoden, gleiche Ausgaben, gleicher Zeit-Nullpunkt (Wanduhrzeit ab
Methodenstart inkl. Startlösung und Modellaufbau):

- `gurobi`: volles MIP (`solve_model`), Verlauf per Gurobi-Callback
- `random_fao`: Random Fix-and-Optimize (`random_fao` in `src/matheuristic.jl`) –
  Baseline für Policy A/B/C. Pro Iteration werden n Reihenfolgevariablen y freigegeben,
  n zufällig aus `n_free_min:n_free_max` (Standard 200:800, fester Wert: min = max),
  gedeckelt auf 75 % aller Maschinenpaare; alle anderen fixiert, dann löst Gurobi das
  Teilproblem. Policies ersetzen nur
  `select_free(inst, incumbent, history, rng, n_free_y)` → Vektor von Maschinenpaaren `(a, b)`
  aus `machine_pairs(inst)`. Baseline `random_free`: zufällige Operationen, deren Paare (nächste
  Nachbarn in der Maschinenreihenfolge zuerst) bis `n_free_y` erreicht ist.
  Weitere Parameter: `sub_time_limit` (Standard 1 s), `accept_equal`, `seed`.
  Bound = statische untere Schranke (`lower_bound`, `src/bounds.jl`).
- Big-M: `build_model(inst; ub = ...)` setzt M = ub (Makespan einer bekannten Lösung) und
  S ≤ ub − p, Cmax ≤ ub. `run_anytime.jl` nutzt für beide Methoden ub = Makespan der Startlösung.

```bash
julia --project=. scripts/run_anytime.jl                          # beide Methoden + Vergleich, DEFAULT_INSTANCES, 60 s
julia --project=. scripts/run_anytime.jl 30 ft10 la16              # eigene Auswahl
julia --project=. scripts/run_anytime.jl --method=random_fao --nfree=200:800 --sub-time=1
julia --project=. scripts/tune_fao.jl --dry-run                   # Grid & Dauer anzeigen
julia --project=. scripts/tune_fao.jl --out=tuning_runde3         # Parameter-Studie (Standard-Grid ≈ 8 min)
julia --project=. scripts/tune_fao.jl --nfree=100,200,400 --subtimes=1,5 ft10 ta41 --seeds=42,43
```

**Referenz: zusammenhängende Teilprobleme (keine Policy).** `window_free`
(`src/reference_neighborhoods.jl`) gibt die y-Paare der Operationen rund um einen zufälligen
Zeitpunkt frei. Damit wird geprüft, ob zusammenhängende Teilprobleme bei gleichem `n_free_y`
deutlich schwerer sind als zufällige. Ergebnisse in `results/tuning_window/` (`results/tuning/`
bleibt unverändert); Vergleich in `comparison.csv` und `median_sub_time.png`:

```bash
julia --project=. scripts/tune_fao.jl --selector=window --nfree=200,400,800,1600 --subtimes=1 --seeds=42
julia --project=. scripts/compare_tuning.jl
```

Ausgabe: `results/gurobi/`, `results/random_fao/` (je Seed: Verlauf, Iterations-Log;
je Instanz: Median-Verlauf, Diagnose, Plot; `summary.csv` mit Mittel/Min/Max über Seeds,
`summary_seeds.csv`), `results/comparison/` (Vergleichsgrafik je Instanz, `summary.csv`,
`overview.png`), `results/tuning/` (Parameter-Studie). Jede Methode schreibt `params.json`.
Kennzahlen: `gap_own_bound` (zur eigenen Schranke) und `primal_gap_bks` (zur besten bekannten Lösung).

**Reproduzierbarkeit:** Jeder Zufall läuft über `MersenneTwister(seed)` mit Seed 42
(`DEFAULT_SEED`); auch Gurobi bekommt `Seed = 42`. Standardmäßig wird nur Seed 42 gerechnet,
weitere per `--seeds`. Da Teilprobleme mit Wanduhr-Zeitlimits gelöst werden, können Läufe auf
anderer Hardware oder unter Last trotzdem leicht abweichen.

**Lange Läufe:** Die Skripte halten den Mac per `caffeinate` wach. Ein Ruhezustand würde
die Wanduhr weiterlaufen lassen und Zeitlimits/Anytime-Kurven verfälschen.

## Nächste Schritte

1. Instanzauswahl für die Experimente aus JSPLIB festlegen
2. `policies.jl`: Policy A/B/C als eigene `select_free`-Funktionen
3. `experiments.jl`: Benchmark mit Logging der Anytime-Performance
