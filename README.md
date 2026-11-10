# demo-bioacustica-abelhas

Demonstração didática de um pipeline de **classificação de sons de abelhas**: síntese de sinais, extração de features espectrais e classificação.

> Todos os sinais são **sintéticos**. Este repositório não contém gravações reais, modelos treinados nem código de projetos de pesquisa ou de terceiros.

## O que faz

O script `bee_sound_demo.py`:

1. sintetiza dois comportamentos (voo, com f0 ≈ 200 Hz, e sonicação em flor, com f0 ≈ 320 Hz) como tons com harmônicos e ruído;
2. extrai features do espectrograma: energia log em 16 bandas de frequência e centroide espectral;
3. treina um Random Forest e avalia com validação cruzada de 5 folds.

## Como executar

```bash
pip install numpy scipy scikit-learn
python bee_sound_demo.py
```

## Sobre o resultado

A acurácia sai em 100%. Isso é esperado e **não é um resultado científico**: os dois comportamentos sintéticos têm frequências bem separadas. O demo serve para mostrar o fluxo (sinal → espectrograma → features → classificador), não para estimar desempenho em campo.

## Para dados reais

- usar gravações reais rotuladas, com divisão treino/teste por **indivíduo ou sessão de gravação**, para evitar vazamento;
- trocar as features por espectrogramas mel e uma CNN;
- avaliar com ruído de ambiente, vento e variação de microfone.

## Autora

Fernanda Neiva Mesquita. [Lattes](https://lattes.cnpq.br/3639678638834402)
# demo-bioacustica-abelhas
Demo: pipeline de classificação de sons de abelhas (features espectrais + Random Forest) com sinais sintéticos
