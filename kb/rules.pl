%% ===========================================================================
%%  KB per il riequilibrio di reti di bike sharing  (rappresentazione a triple)
%%  prop(Individuo, Proprieta, Valore)  -  cfr. dispense cap. 6
%% ===========================================================================
:- multifile prop/3.
:- discontiguous prop/3.
:- dynamic prop/3.

%% ---------- TBox: gerarchia di classi (DAG) --------------------------------
prop(stazione,                 subClassOf, luogo).
prop(zona,                     subClassOf, luogo).
prop(furgone,                  subClassOf, risorsa).
prop(stazione_esportatrice,    subClassOf, stazione).   % di mattina perde bici (area residenziale)
prop(stazione_importatrice,    subClassOf, stazione).   % di mattina accumula bici (area di lavoro / hub)
prop(stazione_equilibrata,     subClassOf, stazione).   % flussi in pareggio
prop(stazione_ricreativa,      subClassOf, stazione_equilibrata). % picchi nelle ore centrali / weekend

%% ereditarieta' del tipo lungo la gerarchia
prop(X, type, C) :- prop(S, subClassOf, C), prop(X, type, S).

%% proprieta' con dominio e codominio (documentazione + controlli di coerenza)
prop(saldo_feriale_mattina,  domain, stazione).   prop(saldo_feriale_mattina,  range, numero).
prop(profilo,                domain, stazione).   prop(profilo,                range, profilo).
prop(zona_di,                domain, stazione).   prop(zona_di,                range, zona).
prop(profilo, functional, true).                  prop(zona_di, functional, true).

%% ---------- conoscenza derivata: classe della stazione dal suo profilo ------
%% (il profilo e' appreso con k-means; la classe e' una proprieta' del profilo
%%  ed e' ereditata da tutte le stazioni che vi appartengono)
prop(S, type, Classe) :- prop(S, profilo, P), prop(P, classe_stazioni, Classe).

%% ---------- fasce orarie ------------------------------------------------------
fascia_ora(H, notte)   :- between(0, 5, H).
fascia_ora(H, mattina) :- between(6, 9, H).
fascia_ora(H, giorno)  :- between(10, 15, H).
fascia_ora(H, sera)    :- between(16, 19, H).
fascia_ora(H, serata)  :- between(20, 23, H).
tipo_giorno(feriale). tipo_giorno(festivo).
fascia(notte). fascia(mattina). fascia(giorno). fascia(sera). fascia(serata).

%% ---------- geometria ---------------------------------------------------------
distanza_km(A, B, D) :-
    prop(A, lat, La), prop(A, lng, Lo), prop(B, lat, Lb), prop(B, lng, Lp),
    P is pi/180,
    DLa is (Lb-La)*P, DLo is (Lp-Lo)*P,
    X is sin(DLa/2)**2 + cos(La*P)*cos(Lb*P)*sin(DLo/2)**2,
    D is 2*6371*asin(min(1.0, sqrt(X))).

vicina(A, B, D) :-
    prop(parametri, raggio_vicinato_km, R),
    prop(B, lat, _), A \== B,
    distanza_km(A, B, D), D =< R.

%% ---------- accesso agli attributi appresi -----------------------------------
saldo(S, Tipo, F, V) :-
    atomic_list_concat([saldo, Tipo, F], '_', P), prop(S, P, V).

%% ---------- livelli di squilibrio (soglie apprese dai dati, per citta') -------
livello_deficit(V, alto)  :- prop(soglie, alto,  T), V =< -T, !.
livello_deficit(V, medio) :- prop(soglie, medio, T), V =< -T.
livello_surplus(V, alto)  :- prop(soglie, alto,  T), V >=  T, !.
livello_surplus(V, medio) :- prop(soglie, medio, T), V >=  T.

%% rischio di svuotamento: la stazione perde piu' bici di quante ne riceve
rischio_svuotamento(S, Tipo, F, Liv) :-
    saldo(S, Tipo, F, V), prop(S, type, stazione), livello_deficit(V, Liv).
%% rischio di saturazione: accumula piu' bici di quante ne perde
rischio_saturazione(S, Tipo, F, Liv) :-
    saldo(S, Tipo, F, V), prop(S, type, stazione), livello_surplus(V, Liv).

%% ---------- ragionamento con negazione per fallimento (default) --------------
%% un deficit e' "compensato" se una stazione vicina ha un surplus: i ciclisti
%% possono ripiegare li' e il problema si attenua da solo
compensata(S, Tipo, F) :-
    rischio_svuotamento(S, Tipo, F, _),
    once(( vicina(S, V, _), rischio_saturazione(V, Tipo, F, _) )).

%% deficit critico: nessuna compensazione nota (assunzione di conoscenza completa)
critica(S, Tipo, F, Liv) :-
    rischio_svuotamento(S, Tipo, F, Liv),
    \+ compensata(S, Tipo, F).

%% ---------- aggregazione sulle zone ------------------------------------------
stazioni_critiche_zona(Z, Tipo, F, L) :-
    prop(Z, type, zona),
    findall(S, (prop(S, zona_di, Z), critica(S, Tipo, F, _)), L).

zona_critica(Z, Tipo, F, N) :-
    stazioni_critiche_zona(Z, Tipo, F, L), length(L, N),
    prop(parametri, min_critiche_zona, M), N >= M.

%% ---------- trasferimenti candidati (donatrice -> ricevente) -------------------
trasferimento(Da, A, Tipo, F, D) :-
    critica(A, Tipo, F, _),
    rischio_saturazione(Da, Tipo, F, _),
    distanza_km(Da, A, D),
    prop(parametri, raggio_furgone_km, R), D =< R.

%% ---------- priorita' di intervento -------------------------------------------
peso_livello(alto, 2.0). peso_livello(medio, 1.0).
peso_classe(stazione_importatrice, 1.2) :- !.
peso_classe(stazione_esportatrice, 1.2) :- !.
peso_classe(_, 1.0).

priorita(S, Tipo, F, P) :-
    critica(S, Tipo, F, Liv), saldo(S, Tipo, F, V),
    peso_livello(Liv, WL),
    ( prop(S, type, C), C \== stazione, prop(C, subClassOf, stazione) -> peso_classe(C, WC) ; WC = 1.0 ),
    P is abs(V) * WL * WC.

%% ---------- varianti per l'ablazione (non usate dal sistema; servono a misurare il contributo di ogni regola) ----------
%% priorita senza la negazione per fallimento: i deficit compensati non vengono esclusi
priorita_senza_compensazione(S, Tipo, F, P) :-
    rischio_svuotamento(S, Tipo, F, Liv), saldo(S, Tipo, F, V),
    peso_livello(Liv, WL),
    ( prop(S, type, C), C \== stazione, prop(C, subClassOf, stazione) -> peso_classe(C, WC) ; WC = 1.0 ),
    P is abs(V) * WL * WC.
%% priorita senza i pesi di livello e di classe: conta solo il saldo, ma restano soglie e negazione per fallimento
priorita_senza_pesi(S, Tipo, F, P) :-
    critica(S, Tipo, F, _), saldo(S, Tipo, F, V), P is abs(V).
%% nessuna regola: solo il saldo medio storico della fascia (tutte le stazioni in deficit)
priorita_solo_saldo(S, Tipo, F, P) :-
    saldo(S, Tipo, F, V), V < 0, P is -V.
