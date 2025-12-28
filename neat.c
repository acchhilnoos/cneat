#include <cstdlib>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DA_INIT_SIZE 16
#define GENOME_NUM_IN 2
#define GENOME_NUM_OUT 1
#define GENOME_INIT_SIZE (GENOME_NUM_IN + GENOME_NUM_OUT) // < DA_INIT_SIZE
#define HASH_INIT_BUCKETS 32
#define HASH_LOAD_THRESHOLD 0.75f

#define HASH(X, Y, H) (((23 * 31 + X) * 31 + Y) % H->cap)

#define DEBUG

#define ACTIVATION_STEPS 5
#define ADD_CONN_MAX_ATTEMPTS 5

#ifndef DEBUG
#define POPULATION_SIZE 16
#define P_MUT_WEIGHTS 0.8f
#define P_MUT_ADD_CONN 0.05f
#define P_MUT_ADD_NODE 0.03f
#else
#define POPULATION_SIZE 5
#define P_MUT_WEIGHTS 1.0f
#define P_MUT_ADD_CONN 1.0f
#define P_MUT_ADD_NODE 1.0f
#endif

#define DEFINE_STRUCT_DA(TYPE)                                                 \
  typedef struct {                                                             \
    struct TYPE *data;                                                         \
    size_t size, cap;                                                          \
  } TYPE##_DA

#define DA_AT(DA, AT) DA->data[AT]
#define DA_ADD(DA, EL, TYPE)                                                   \
  if (DA->size + 1 >= DA->cap) {                                               \
    DA->cap *= 2;                                                              \
    DA->data = realloc(DA->data, DA->cap * sizeof(*DA->data));                 \
    if (!DA->data)                                                             \
      exit(EXIT_FAILURE);                                                      \
  }                                                                            \
  struct TYPE *EL = &DA_AT(DA, DA->size++)
#define DA_FREE(DA)                                                            \
  free(DA->data);                                                              \
  free(DA)

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
  Conn_DA *conns;
  Node_DA *nodes;
  float fitness;
};

struct Entry {
  size_t in, out, id;
  struct Entry *next;
};

struct Hashtbl {
  struct Entry **entries;
  size_t size, cap;
};

struct Population {
  struct Genome *genomes;
  struct Hashtbl *history;
};

// hashtbl {{{
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
        size_t hash = HASH(e->in, e->out, h);
        e->next = h->entries[hash];
        h->entries[hash] = e;
        e = next;
      }
    }
    free(old);
  }

  size_t hash = HASH(in, out, h);
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
Conn_DA *init_conns() {
  Conn_DA *c = malloc(sizeof(*c));
  if (!c)
    exit(EXIT_FAILURE);
  c->data = calloc(DA_INIT_SIZE, sizeof(*c->data));
  if (!c->data)
    exit(EXIT_FAILURE);
  c->size = 0;
  c->cap = DA_INIT_SIZE;
  return c;
}

Node_DA *init_nodes() {
  Node_DA *n = malloc(sizeof(*n));
  if (!n)
    exit(EXIT_FAILURE);
  n->data = calloc(DA_INIT_SIZE, sizeof(*n->data));
  if (!n->data)
    exit(EXIT_FAILURE);
  for (int i = 0; i < GENOME_INIT_SIZE; i++) {
    DA_AT(n, i).id = i;
    DA_AT(n, i).type = i < GENOME_NUM_IN ? INPUT : OUTPUT;
    DA_AT(n, i).value = 0.0f;
  }
  n->size = GENOME_INIT_SIZE;
  n->cap = DA_INIT_SIZE;
  return n;
}

struct Genome *init_genomes() {
  struct Genome *g = calloc(POPULATION_SIZE, sizeof(*g));
  if (!g)
    exit(EXIT_FAILURE);
  for (int i = 0; i < POPULATION_SIZE; i++) {
    g[i].conns = init_conns();
    if (!g[i].conns)
      exit(EXIT_FAILURE);
    g[i].nodes = init_nodes();
    if (!g[i].nodes)
      exit(EXIT_FAILURE);
    g[i].fitness = 0.0f;
  }
  return g;
}

struct Population *init_population() {
  struct Population *p = malloc(sizeof(*p));
  if (!p)
    exit(EXIT_FAILURE);
  p->genomes = init_genomes();
  if (!p->genomes)
    exit(EXIT_FAILURE);
  p->history = init_hashtbl();
  if (!p->history)
    exit(EXIT_FAILURE);
  return p;
}
// }}}

// dump {{{
void dump(struct Population *p) {
  for (int i = 0; i < POPULATION_SIZE; i++) {
    printf("Genome %d:\n", i);
    struct Genome *g = &p->genomes[i];
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
      printf("    [%2zu] %2zu %3zu | %7.3f\n", c->id, c->in, c->out, c->weight);
    }
  }
}
// }}}

// wipe {{{
void wipe(struct Population *p) {
  for (int i = 0; i < POPULATION_SIZE; i++) {
    DA_FREE(p->genomes[i].conns);
    DA_FREE(p->genomes[i].nodes);
  }
  free(p->genomes);
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
  free(p);
}
// }}}

static inline void copy_conn(struct Conn *a, struct Conn *b) {
  b->in = a->in;
  b->out = a->out;
  b->id = a->id;
  b->weight = a->weight;
  b->enabled = a->enabled;
}

int compare_conn_id(const void *a, const void *b) {
  const struct Conn *c1 = (const struct Conn *)a;
  const struct Conn *c2 = (const struct Conn *)b;
  if (c1->id < c2->id)
    return -1;
  if (c1->id > c2->id)
    return 1;
  return 0;
}

struct Genome *crossover(struct Genome *p1, struct Genome *p2) {
  struct Genome *child = malloc(sizeof(*child));
  if (!child)
    exit(EXIT_FAILURE);
  child->conns = init_conns();
  child->nodes = init_nodes();
  child->nodes->size = 0;

  struct Genome *mfit, *lfit;
  if (p1->fitness > p2->fitness) {
    mfit = p1;
    lfit = p2;
  } else if (p2->fitness > p1->fitness) {
    mfit = p2;
    lfit = p1;
  } else {
    mfit = (random() % 2 == 0) ? p1 : p2;
    lfit = (mfit == p1) ? p2 : p1;
  }

  qsort(p1->conns->data, p1->conns->size, sizeof(struct Conn), compare_conn_id);
  qsort(p2->conns->data, p2->conns->size, sizeof(struct Conn), compare_conn_id);

  size_t i = 0, j = 0;
  while (i < mfit->conns->size && j < lfit->conns->size) {
    struct Conn *ci = &DA_AT(mfit->conns, i);
    struct Conn *cj = &DA_AT(lfit->conns, j);

    DA_ADD(child->conns, c, Conn);
    // struct Conn *c = &DA_AT(child->conns, child->conns->size++);

    if (ci->id == cj->id) {
      struct Conn *chosen = (random() % 2 == 0) ? ci : cj;
      copy_conn(chosen, c);
      if (!ci->enabled || !cj->enabled) {
        if (frand1() < 0.75)
          c->enabled = FALSE;
      }
      i++;
      j++;
    } else if (ci->id < cj->id) {
      copy_conn(ci, c);
      i++;
    } else {
      // child->conns->size--; ?????
      j++;
    }
  }
  while (i < mfit->conns->size) {
    DA_ADD(child->conns, c, Conn);
    // struct Conn *c = &DA_AT(child->conns, child->conns->size++);
    copy_conn(&DA_AT(mfit->conns, i), c);
    i++;
  }

  size_t max_id = 0;
  for (i = 0; i < child->conns->size; i++) {
    if (DA_AT(child->conns, i).in > max_id)
      max_id = DA_AT(child->conns, i).in;
    if (DA_AT(child->conns, i).out > max_id)
      max_id = DA_AT(child->conns, i).out;
  }
  char *nodes_added = calloc(max_id + 1, sizeof(char));
  if (!nodes_added)
    exit(EXIT_FAILURE);

  for (i = 0; i < GENOME_INIT_SIZE; i++) {
    if (!nodes_added[i]) {
      DA_ADD(child->nodes, n, Node);
      // struct Node *n = &DA_AT(child->nodes, child->nodes->size++);
      n->id = i;
      n->type = i < GENOME_NUM_IN ? INPUT : OUTPUT;
      nodes_added[i] = 1;
    }
  }

  for (i = 0; i < child->conns->size; i++) {
    struct Conn *c = &DA_AT(child->conns, i);
    if (!nodes_added[c->in]) {
      DA_ADD(child->nodes, n, Node);
      // struct Node *n = &DA_AT(child->nodes, child->nodes->size++);
      n->id = c->in;
      n->type = HIDDEN;
      nodes_added[c->in] = 1;
    }
    if (!nodes_added[c->out]) {
      DA_ADD(child->nodes, n, Node);
      // struct Node *n = &DA_AT(child->nodes, child->nodes->size++);
      n->id = c->out;
      n->type = HIDDEN;
      nodes_added[c->out] = 1;
    }
  }

  free(nodes_added);
  return child;
}

// mutation {{{
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

    DA_ADD(g->conns, c, Conn);
    // if (g->conns->size + 1 >= g->conns->cap) {
    //   g->conns->cap *= 2;
    //   g->conns->data =
    //       realloc(g->conns->data, g->conns->cap * sizeof(*g->conns->data));
    //   if (!g->conns->data)
    //     exit(EXIT_FAILURE);
    // }
    // struct Conn *c = &DA_AT(g->conns, g->conns->size++);
    c->in = in->id;
    c->out = out->id;
    c->weight = 1.0f;
    c->enabled = TRUE;
    c->id = ht_insert(h, c->in, c->out);
    return;
  }
}

void mut_add_node(struct Genome *g, struct Hashtbl *h) {
  if (frand1() > P_MUT_ADD_NODE)
    return;
  if (g->conns->size == 0)
    return;
  DA_ADD(g->nodes, n, Node);
  // if (g->nodes->size + 1 >= g->nodes->cap) {
  //   g->nodes->cap *= 2;
  //   g->nodes->data =
  //       realloc(g->nodes->data, g->nodes->cap * sizeof(*g->nodes->data));
  //   if (!g->nodes->data)
  //     exit(EXIT_FAILURE);
  // }
  // struct Node *n = &DA_AT(g->nodes, g->nodes->size++);
  n->id = next_node_id++;
  n->type = HIDDEN;
  n->value = 0.0f;
  struct Conn *c = &DA_AT(g->conns, random() % g->conns->size);
  c->enabled = FALSE;
  DA_ADD(g->conns, n1, Conn);
  DA_ADD(g->conns, n2, Conn);
  // if (g->conns->size + 2 >= g->conns->cap) {
  //   g->conns->cap *= 2;
  //   g->conns->data =
  //       realloc(g->conns->data, g->conns->cap * sizeof(*g->conns->data));
  //   if (!g->conns->data)
  //     exit(EXIT_FAILURE);
  // }
  // struct Conn *n1 = &DA_AT(g->conns, g->conns->size++);
  // struct Conn *n2 = &DA_AT(g->conns, g->conns->size++);
  n1->in = c->in;
  n1->out = n->id;
  n1->weight = 1.0f;
  n1->enabled = TRUE;
  n1->id = ht_insert(h, n1->in, n1->out);
  n2->in = n->id;
  n2->out = c->out;
  n2->weight = c->weight;
  n2->enabled = TRUE;
  n2->id = ht_insert(h, n2->in, n2->out);
}

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
  for (int i = 0; i < POPULATION_SIZE; i++) {
    struct Genome *g = &p->genomes[i];
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
  mutate(p);
  mutate(p);
  dump(p);

  wipe(p);

  return EXIT_SUCCESS;
}
