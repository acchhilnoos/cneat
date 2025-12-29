#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ACTIVATION_STEPS 5
#define ADD_CONN_MAX_ATTEMPTS 5
#define DA_INIT_SIZE 16
#define GENOME_NUM_IN 2
#define GENOME_NUM_OUT 1
#define GENOME_INIT_SIZE (GENOME_NUM_IN + GENOME_NUM_OUT)
#define HASH_INIT_BUCKETS 32
#define HASH_LOAD_THRESHOLD 0.75f

#define DEBUG

#ifndef DEBUG
#define POPULATION_SIZE 32
#define P_MUT_ADD_CONN 0.05f
#define P_MUT_ADD_NODE 0.03f
#define P_MUT_WEIGHTS 0.8f
#else
#define POPULATION_SIZE 3
#define P_MUT_ADD_CONN 1.0f
#define P_MUT_ADD_NODE 1.0f
#define P_MUT_WEIGHTS 1.0f
#endif

#define DA_ADD(DA, EL)                                                         \
  if (DA->size + 1 >= DA->cap) {                                               \
    DA->cap *= 2;                                                              \
    DA->data = realloc(DA->data, DA->cap * sizeof(*DA->data));                 \
    if (!DA->data)                                                             \
      exit(EXIT_FAILURE);                                                      \
  }                                                                            \
  EL = &DA_AT(DA, DA->size++)
#define DA_AT(DA, AT) DA->data[AT]
#define DA_FREE(DA)                                                            \
  free(DA->data);                                                              \
  free(DA)
#define DA_INIT(DA, N)                                                         \
  DA = malloc(sizeof(*DA));                                                    \
  if (!DA)                                                                     \
    exit(EXIT_FAILURE);                                                        \
  DA->data = calloc(N, sizeof(*DA->data));                                     \
  if (!DA->data)                                                               \
    exit(EXIT_FAILURE);                                                        \
  DA->size = 0;                                                                \
  DA->cap = N
#define DEFINE_STRUCT_DA(TYPE)                                                 \
  typedef struct {                                                             \
    struct TYPE *data;                                                         \
    size_t size, cap;                                                          \
  } TYPE##_DA

static inline float frand(float max) {
  return ((float)(random() % 1000)) * max / 1000.0f;
}
static inline float frand1() { return ((float)(random() % 1000)) / 1000.0f; }

static size_t next_conn_id = 0;
struct Conn {
  size_t in, out, id;
  float weight;
  enum { FALSE, TRUE } enabled;
};
DEFINE_STRUCT_DA(Conn);

static size_t next_node_id = GENOME_INIT_SIZE;
struct Node {
  size_t id;
  enum { INPUT, BIAS, HIDDEN, OUTPUT } type;
  float value;
};
DEFINE_STRUCT_DA(Node);

struct Genome {
  size_t id;
  Conn_DA *conns;
  Node_DA *nodes;
  float fitness;
};
DEFINE_STRUCT_DA(Genome);

struct Species {
  size_t id;
  size_t staleness;
  struct Genome *representative;
  Genome_DA *genomes;
  float species_total_fitness;
};
DEFINE_STRUCT_DA(Species);

struct Entry {
  size_t in, out, id;
  struct Entry *next;
};

struct Population {
  Genome_DA *genomes;
  Species_DA *species;
  struct Hashtbl *history;
  size_t next_genome_id;
  size_t next_species_id;

  float C1, C2, C3;
  float compatibility_threshold;
};

// hashtbl {{{

struct Hashtbl {
  struct Entry **entries;
  size_t size, cap;
};
static inline size_t hash_func(size_t in, size_t out, struct Hashtbl *h) {
  return (((23 * 31 + in) * 31 + out) % h->cap);
}

struct Hashtbl *init_hashtbl() {
  struct Hashtbl *h = malloc(sizeof(*h));
  if (!h)
    exit(EXIT_FAILURE);
  h->cap = HASH_INIT_BUCKETS;
  h->entries = calloc(h->cap, sizeof(*h->entries));
  if (!h->entries)
    exit(EXIT_FAILURE);
  h->size = 0;
  return h;
}

int ht_insert(struct Hashtbl *h, size_t in, size_t out) {
  if ((float)h->size / h->cap > HASH_LOAD_THRESHOLD) {
    struct Entry **old = h->entries;
    size_t old_cap = h->cap;
    h->cap *= 2;

    h->entries = calloc(h->cap, sizeof(struct Entry *));
    if (!h->entries)
      exit(EXIT_FAILURE);

    for (size_t i = 0; i < old_cap; i++) {
      struct Entry *e = old[i];
      while (e) {
        struct Entry *next = e->next;
        size_t hash = hash_func(e->in, e->out, h);
        e->next = h->entries[hash];
        h->entries[hash] = e;
        e = next;
      }
    }
    free(old);
  }

  size_t hash = hash_func(in, out, h);
  struct Entry **e = &h->entries[hash];
  while (*e) {
    if ((*e)->in == in && (*e)->out == out)
      return (*e)->id;
    e = &(*e)->next;
  }
  *e = malloc(sizeof(**e));
  if (!*e)
    exit(EXIT_FAILURE);
  (*e)->in = in;
  (*e)->out = out;
  (*e)->id = next_conn_id++;
  (*e)->next = NULL;
  h->size++;
  return (*e)->id;
}
// }}}

// init_* {{{

Conn_DA *init_conns(struct Hashtbl *h) {
  Conn_DA *c;
  DA_INIT(c, DA_INIT_SIZE);
  for (int i = 0; i < GENOME_NUM_IN; i++) {
    for (int j = GENOME_NUM_IN; j < GENOME_INIT_SIZE; j++) {
      struct Conn *t;
      DA_ADD(c, t);
      t->in = i;
      t->out = j;
      t->id = ht_insert(h, i, j);
      t->weight = 1.0f;
      t->enabled = TRUE;
    }
  }
  return c;
}

Node_DA *init_nodes() {
  Node_DA *n;
  DA_INIT(n, DA_INIT_SIZE);
  for (int i = 0; i < GENOME_INIT_SIZE; i++) {
    struct Node *t;
    DA_ADD(n, t);
    t->id = i;
    t->type = i < GENOME_NUM_IN ? INPUT : OUTPUT;
    t->value = 0.0f;
  }
  return n;
}

struct Population *init_population() {
  struct Population *p = malloc(sizeof(*p));
  if (!p)
    exit(EXIT_FAILURE);

  p->history = init_hashtbl();
  if (!p->history)
    exit(EXIT_FAILURE);

  DA_INIT(p->genomes, POPULATION_SIZE);
  p->next_genome_id = 0;
  for (int i = 0; i < POPULATION_SIZE; i++) {
    struct Genome *g = &DA_AT(p->genomes, i);
    g->id = p->next_genome_id++;
    g->conns = init_conns(p->history);
    if (!g->conns)
      exit(EXIT_FAILURE);
    g->nodes = init_nodes();
    if (!g->nodes)
      exit(EXIT_FAILURE);
    g->fitness = 0.0f;
  }
  p->genomes->size = POPULATION_SIZE;

  DA_INIT(p->species, DA_INIT_SIZE);
  p->next_species_id = 0;
  p->C1 = 1.0f; // Disjoint coefficient
  p->C2 = 1.0f; // Excess coefficient
  p->C3 = 0.4f; // Weight difference coefficient
  p->compatibility_threshold = 3.0f;
  return p;
}
// }}}

// dump {{{

void dump(struct Population *p) {
  for (int i = 0; i < p->genomes->size; i++) {
    struct Genome *g = &DA_AT(p->genomes, i);
    printf("Genome %zu:\n", g->id);
    printf("  Nodes: %zu\n", g->nodes->size);
    for (int j = 0; j < g->nodes->size; j++) {
      struct Node *n = &DA_AT(g->nodes, j);
      printf("    [%2zu] %s | %7.3f\n", n->id,
             n->type == 0   ? "INPUT "
             : n->type == 1 ? "BIAS  "
             : n->type == 2 ? "HIDDEN"
             : n->type == 3 ? "OUTPUT"
                            : "???   ",
             n->value);
    }
    printf("  Connections: %zu\n", g->conns->size);
    for (int j = 0; j < g->conns->size; j++) {
      struct Conn *c = &DA_AT(g->conns, j);
      printf("  %c [%2zu] %2zu %3zu | %7.3f\n", c->enabled ? ' ' : 'x', c->id,
             c->in, c->out, c->weight);
    }
  }
}
// }}}

// wipe {{{

void free_genomes(struct Population *p) {
  for (int i = 0; i < p->genomes->size; i++) {
    DA_FREE(DA_AT(p->genomes, i).conns);
    DA_FREE(DA_AT(p->genomes, i).nodes);
  }
  DA_FREE(p->genomes);
}

static inline void free_species(struct Population *p) {
  for (int i = 0; i < p->species->size; i++) {
    DA_FREE(DA_AT(p->species, i).genomes);
  }
  DA_FREE(p->species);
}

void free_history(struct Population *p) {
  for (size_t i = 0; i < p->history->cap; i++) {
    struct Entry *c = p->history->entries[i];
    while (c) {
      struct Entry *n = c->next;
      free(c);
      c = n;
    }
  }
  free(p->history->entries);
  free(p->history);
}

void wipe(struct Population *p) {
  free_genomes(p);
  free_history(p);
  free_species(p);
  free(p);
}
// }}}

// crossover {{{

static inline void copy_conn(struct Conn *a, struct Conn *b) {
  b->in = a->in;
  b->out = a->out;
  b->id = a->id;
  b->weight = a->weight;
  b->enabled = a->enabled;
}

int compare_conn(const void *a, const void *b) {
  const struct Conn *c1 = (const struct Conn *)a;
  const struct Conn *c2 = (const struct Conn *)b;
  if (c1->id < c2->id)
    return -1;
  if (c1->id > c2->id)
    return 1;
  return 0;
}

int compare_genome(const void *a, const void *b) {
  const struct Genome *g1 = *(const struct Genome **)a;
  const struct Genome *g2 = *(const struct Genome **)b;
  if (g1->fitness > g2->fitness)
    return -1;
  if (g1->fitness < g2->fitness)
    return 1;
  return 0;
}

void crossover(struct Genome *g1, struct Genome *g2, struct Genome *child,
               struct Population *p) {
  child->id = p->next_genome_id++;
  child->conns = init_conns(p->history);
  child->conns->size = 0;
  child->nodes = init_nodes();
  child->nodes->size = 0;
  child->fitness = 0.0f;

  struct Genome *mfit, *lfit;
  if (g1->fitness > g2->fitness) {
    mfit = g1;
    lfit = g2;
  } else if (g2->fitness > g1->fitness) {
    mfit = g2;
    lfit = g1;
  } else {
    mfit = (random() % 2 == 0) ? g1 : g2;
    lfit = (mfit == g1) ? g2 : g1;
  }

  qsort(g1->conns->data, g1->conns->size, sizeof(struct Conn), compare_conn);
  qsort(g2->conns->data, g2->conns->size, sizeof(struct Conn), compare_conn);

  size_t i = 0, j = 0, max_id = GENOME_INIT_SIZE - 1;
  while (i < mfit->conns->size && j < lfit->conns->size) {
    struct Conn *ci = &DA_AT(mfit->conns, i);
    struct Conn *cj = &DA_AT(lfit->conns, j);

    if (ci->id == cj->id) {
      struct Conn *chosen = (random() % 2 == 0) ? ci : cj;
      struct Conn *c;
      DA_ADD(child->conns, c);
      copy_conn(chosen, c);
      if (!ci->enabled || !cj->enabled) {
        if (frand1() < 0.75)
          c->enabled = FALSE;
      }
      if (ci->in > max_id)
        max_id = ci->in;
      if (ci->out > max_id)
        max_id = ci->out;
      i++;
      j++;
    } else if (ci->id < cj->id) {
      struct Conn *c;
      DA_ADD(child->conns, c);
      copy_conn(ci, c);
      if (ci->in > max_id)
        max_id = ci->in;
      if (ci->out > max_id)
        max_id = ci->out;
      i++;
    } else {
      j++;
    }
  }
  while (i < mfit->conns->size) {
    struct Conn *c;
    DA_ADD(child->conns, c);
    copy_conn(&DA_AT(mfit->conns, i), c);
    if (DA_AT(mfit->conns, i).in > max_id)
      max_id = DA_AT(mfit->conns, i).in;
    if (DA_AT(mfit->conns, i).out > max_id)
      max_id = DA_AT(mfit->conns, i).out;
    i++;
  }

  char *added = calloc(max_id + 1, sizeof(char));
  if (!added)
    exit(EXIT_FAILURE);

  for (i = 0; i < GENOME_INIT_SIZE; i++) {
    if (!added[i]) {
      struct Node *n;
      DA_ADD(child->nodes, n);
      n->id = i;
      n->type = i < GENOME_NUM_IN ? INPUT : OUTPUT;
      added[i] = 1;
    }
  }

  for (i = 0; i < child->conns->size; i++) {
    struct Conn *c = &DA_AT(child->conns, i);
    if (!added[c->in]) {
      struct Node *n;
      DA_ADD(child->nodes, n);
      n->id = c->in;
      n->type = HIDDEN;
      added[c->in] = 1;
    }
    if (!added[c->out]) {
      struct Node *n;
      DA_ADD(child->nodes, n);
      n->id = c->out;
      n->type = HIDDEN;
      added[c->out] = 1;
    }
  }
  free(added);
}
// }}}

// speciation {{{

float compatibility(struct Genome *g1, struct Genome *g2, float C1, float C2,
                    float C3) {
  size_t i = 0, j = 0, e = 0, d = 0, m = 0;
  float w = 0.0;

  // TODO: ensure sorted
  size_t max_innov_g1 =
      (g1->conns->size > 0) ? DA_AT(g1->conns, g1->conns->size - 1).id : 0;
  size_t max_innov_g2 =
      (g2->conns->size > 0) ? DA_AT(g2->conns, g2->conns->size - 1).id : 0;

  while (i < g1->conns->size && j < g2->conns->size) {
    struct Conn *ci = &DA_AT(g1->conns, i);
    struct Conn *cj = &DA_AT(g2->conns, j);

    if (ci->id == cj->id) {
      m++;
      w += fabs(ci->weight - cj->weight);
      i++;
      j++;
    } else if (ci->id < cj->id) {
      d++;
      i++;
    } else {
      d++;
      j++;
    }
  }
  while (i < g1->conns->size) {
    e++;
    i++;
  }
  while (j < g2->conns->size) {
    e++;
    j++;
  }

  size_t N =
      (g1->conns->size > g2->conns->size) ? g1->conns->size : g2->conns->size;
  if (N < 20)
    N = 1;

  float distance = (C1 * e / N) + (C2 * d / N);

  if (m > 0)
    distance += (C3 * w / m);

  return distance;
}

void speciate(struct Population *p) {
  for (int i = 0; i < p->species->size; i++)
    DA_AT(p->species, i).genomes->size = 0;

  for (int i = 0; i < p->genomes->size; i++) {
    struct Genome *g = &DA_AT(p->genomes, i);

    char found = 0;
    for (int j = 0; j < p->species->size; j++) {
      struct Species *s = &DA_AT(p->species, j);
      float d = compatibility(g, s->representative, p->C1, p->C2, p->C3);

      if (d < p->compatibility_threshold) {
        // TODO: not DA_ADD
        DA_ADD(s->genomes, g);
        found = 1;
        break;
      }
    }
    if (found)
      continue;

    struct Species *new_s;
    DA_ADD(p->species, new_s);
    new_s->id = p->next_species_id++;
    new_s->staleness = 0;
    new_s->representative = g;
    DA_INIT(new_s->genomes, POPULATION_SIZE);
    // TODO: not DA_ADD
    DA_ADD(new_s->genomes, g);
    new_s->species_total_fitness = 0.0f;
  }

  size_t valid = 0;
  for (int i = 0; i < p->species->size; i++) {
    struct Species *s = &DA_AT(p->species, i);
    if (s->genomes->size > 0) {
      if (valid != i)
        DA_AT(p->species, i) = *s;
      valid++;
    } else {
      DA_FREE(s->genomes);
    }
  }
  p->species->size = valid;
}
// }}}

void share_species_fitness(struct Population *p) {
  for (int i = 0; i < p->species->size; i++) {
    struct Species *s = &DA_AT(p->species, i);
    s->species_total_fitness = 0.0f;

    // if (s->genomes->size == 0)
    //   continue;

    for (int j = 0; j < s->genomes->size; j++) {
      struct Genome *g = &DA_AT(s->genomes, j);
      g->fitness /= s->genomes->size;
      s->species_total_fitness += g->fitness;
    }
  }
}

void evaluate(struct Population *p) {
  // TODO: fitness
  for (int i = 0; i < p->genomes->size; i++) {
    struct Genome *g = &DA_AT(p->genomes, i);
    g->fitness = g->conns->size;
  }
}

// reproduction {{{

void reproduce(struct Population *p) {
  // TODO: evaluate, speciate, reproduce, mutate

  speciate(p);
  share_species_fitness(p);

  Genome_DA *next_genomes;
  DA_INIT(next_genomes, POPULATION_SIZE);

  float population_total_fitness = 0.0f;
  for (int i = 0; i < p->species->size; i++) {
    population_total_fitness += DA_AT(p->species, i).species_total_fitness;
  }

  if (population_total_fitness == 0.0f) {
    for (int i = 0; i < POPULATION_SIZE; ++i) {
      struct Genome *old_g = &DA_AT(p->genomes, i % p->genomes->size);
      struct Genome *new_g;
      DA_ADD(next_genomes, new_g);
      // TODO: Deep copy old_g to new_g, and assign new ID
      crossover(&DA_AT(p->genomes, 0), &DA_AT(p->genomes, 1), new_g, p);
    }
  } else {
    for (int i = 0; i < p->species->size; i++) {
      struct Species *s = &DA_AT(p->species, i);
      qsort(s->genomes->data, s->genomes->size, sizeof(struct Genome *),
            compare_genome);
    }

    for (int i = 0; i < p->species->size; i++) {
      struct Species *s = &DA_AT(p->species, i);
      int num_offspring =
          (int)roundf(s->species_total_fitness / population_total_fitness *
                      POPULATION_SIZE);

      if (num_offspring > 0) {
        struct Genome *champion = &DA_AT(s->genomes, 0);
        struct Genome *new_g;
        DA_ADD(next_genomes, new_g);
        // TODO: Deep copy champion to new_g, and assign new ID
        crossover(champion, champion, new_g, p);
        num_offspring--;
      }

      // TODO: cull
      int num_parents = s->genomes->size;

      for (int k = 0; k < num_offspring; k++) {
        struct Genome *parent1 = &DA_AT(s->genomes, random() % num_parents);
        struct Genome *parent2 = &DA_AT(s->genomes, random() % num_parents);
        struct Genome *child;
        // TODO: overpop check
        DA_ADD(next_genomes, child);
        crossover(parent1, parent2, child, p);
        // TODO: Apply mutation here after crossover
      }
    }
  }

  while (next_genomes->size < POPULATION_SIZE) {
    struct Genome *new_g;
    DA_ADD(next_genomes, new_g);
    // Clone a random existing genome, or just use a dummy for now
    crossover(&DA_AT(p->genomes, 0), &DA_AT(p->genomes, 1), new_g, p);
  }

  free_genomes(p);

  p->genomes = next_genomes;
}
// }}}

// mutation {{{

// Mutates the connection weights of a genome. Each weight has a chance to be
// uniformly perturbed or assigned a new random value.
void mutate_weights(struct Genome *g) {
  if (frand1() > P_MUT_WEIGHTS)
    return;
  for (int i = 0; i < g->conns->size; i++) {
    if (random() % 10 == 0) {
      DA_AT(g->conns, i).weight = frand(2.0f) - 1.0f;
    } else {
      float t = DA_AT(g->conns, i).weight + frand1() - 0.5f;
      t = t > 1.0f ? 1.0f : t < -1.0f ? -1.0f : t;
      DA_AT(g->conns, i).weight = t;
    }
  }
}

// Adds a new connection between two existing nodes in a genome if one does not
// already exist. The innovation is recorded in the history table.
void mut_add_conn(struct Genome *g, struct Hashtbl *h) {
  if (frand1() > P_MUT_ADD_CONN)
    return;
  for (int i = 0; i < ADD_CONN_MAX_ATTEMPTS; i++) {
    struct Node *in;
    struct Node *out;
    do {
      in = &DA_AT(g->nodes, random() % g->nodes->size);
    } while (in->type == OUTPUT);
    do {
      out = &DA_AT(g->nodes, random() % g->nodes->size);
    } while (out->type == INPUT || out == in);

    char exists = 0;
    for (int j = 0; j < g->conns->size; j++) {
      struct Conn *c = &DA_AT(g->conns, j);
      if (c->in == in->id && c->out == out->id ||
          c->in == out->id && c->out == in->id) {
        exists = 1;
        break;
      }
    }
    if (exists)
      continue;

    struct Conn *c;
    DA_ADD(g->conns, c);
    c->in = in->id;
    c->out = out->id;
    c->weight = 1.0f;
    c->enabled = TRUE;
    c->id = ht_insert(h, c->in, c->out);
    return;
  }
}

// Adds a new node to a genome by splitting an existing connection. The old
// connection is disabled, and two new ones are created.
void mut_add_node(struct Genome *g, struct Hashtbl *h) {
  if (frand1() > P_MUT_ADD_NODE)
    return;
  if (g->conns->size == 0)
    return;

  struct Node *n;
  DA_ADD(g->nodes, n);
  n->id = next_node_id++;
  n->type = HIDDEN;
  n->value = 0.0f;
  struct Conn *c = &DA_AT(g->conns, random() % g->conns->size);
  c->enabled = FALSE;

  struct Conn *n1;
  DA_ADD(g->conns, n1);
  n1->in = c->in;
  n1->out = n->id;
  n1->weight = 1.0f;
  n1->enabled = TRUE;
  n1->id = ht_insert(h, n1->in, n1->out);

  struct Conn *n2;
  DA_ADD(g->conns, n2);
  n2->in = n->id;
  n2->out = c->out;
  n2->weight = c->weight;
  n2->enabled = TRUE;
  n2->id = ht_insert(h, n2->in, n2->out);
}

// Applies all forms of mutation (weights, add connection, add node) to each
// genome in the population.
void mutate(struct Population *p) {
  /*
   * There was an 80% chance of a genome having its connection weights
   * mutated, in which case each weight had a 90% chance of being uniformly
   * perturbed and a 10% chance of being assigned a new random value. There
   * was a 75% chance that an inherited gene was disabled if it was disabled
   * in either parent. In each generation, 25% of offspring resulted from
   * mutation without crossover. The interspecies mating rate was 0.001. In
   * smaller populations, the probability of adding a new node was 0.03 and
   * the probability of a new link mutation was 0.05.
   */
  struct Hashtbl *h = p->history;
  for (int i = 0; i < p->genomes->size; i++) {
    struct Genome *g = &DA_AT(p->genomes, i);
    mutate_weights(g);
    mut_add_conn(g, h);
    mut_add_node(g, h);
  }
}
// }}}

int main(int argc, char *argv[]) {
  // srandom(time(NULL));
  srandom(1);
  struct Population *p = init_population();

  dump(p);
  mutate(p);
  evaluate(p);
  reproduce(p);
  mutate(p);
  // dump(p);
  // mutate(p);
  evaluate(p);
  reproduce(p);
  dump(p);

  wipe(p);

  return EXIT_SUCCESS;
}
