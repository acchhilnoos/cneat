#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEBUG

#define DA_INIT_SIZE 16
#define GENOME_NUM_IN 2
#define GENOME_NUM_OUT 1
#define GENOME_INIT_SIZE (GENOME_NUM_IN + GENOME_NUM_OUT) // < DA_INIT_SIZE
#define HASH_INIT_BUCKETS 32
#define HASH_LOAD_THRESHOLD 0.75f

#define HASH(X, Y, H) (((23 * 31 + X) * 31 + Y) % H->cap)

#ifndef DEBUG
#define POPULATION_SIZE 16
#define P_MUT_WEIGHTS 0.8f
#define P_MUT_ADD_CONN 0.05f
#define P_MUT_ADD_NODE 0.03f
#else
#define POPULATION_SIZE 1
#define P_MUT_WEIGHTS 1.0f
#define P_MUT_ADD_CONN 1.0f
#define P_MUT_ADD_NODE 1.0f
#endif

#define MUT_MAX_ATTEMPTS 5

#define DEFINE_STRUCT_DA(TYPE)                                                 \
  typedef struct {                                                             \
    struct TYPE *data;                                                         \
    size_t size, cap;                                                          \
  } TYPE##_DA

#define DA_AT(DA, AT) DA->data[AT]
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

static size_t next_node_id = 0;
struct Node {
  size_t id;
  enum { INPUT, BIAS, HIDDEN, OUTPUT } type;
  float value;
};
DEFINE_STRUCT_DA(Node);

struct Genome {
  Conn_DA *conns;
  Node_DA *nodes;
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

struct Hashtbl *init_hashtbl() {
  struct Hashtbl *h = malloc(sizeof(*h));
  if (!h)
    return NULL;
  h->cap = HASH_INIT_BUCKETS;
  h->entries = calloc(h->cap, sizeof(*h->entries));
  if (!h->entries) {
    free(h);
    return NULL;
  }
  h->size = 0;
  return h;
}

int ht_insert(struct Hashtbl *h, size_t in, size_t out) {
  if ((float)h->size / h->cap > HASH_LOAD_THRESHOLD) {
    struct Entry **old = h->entries;
    size_t old_cap = h->cap;
    h->cap *= 2;

    h->entries = calloc(h->cap, sizeof(struct Entry *));
    if (!h->entries) {
      h->entries = old;
      fprintf(stderr, "Error: Rehashing failed due to memory allocation.\n");
      return -1;
    }

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
    return -1;
  (*e)->in = in;
  (*e)->out = out;
  (*e)->id = next_conn_id++;
  (*e)->next = NULL;
  h->size++;
  return (*e)->id;
}

Conn_DA *init_conns() {
  Conn_DA *c = malloc(sizeof(*c));
  if (!c)
    return NULL;
  c->data = calloc(DA_INIT_SIZE, sizeof(*c->data));
  if (!c->data) {
    free(c);
    return NULL;
  }
  c->size = 0;
  c->cap = DA_INIT_SIZE;
  return c;
}

Node_DA *init_nodes() {
  Node_DA *n = malloc(sizeof(*n));
  if (!n)
    return NULL;
  n->data = calloc(DA_INIT_SIZE, sizeof(*n->data));
  if (!n->data) {
    free(n);
    return NULL;
  }
  for (int i = 0; i < GENOME_INIT_SIZE; i++) {
    DA_AT(n, i).id = next_node_id++;
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
    return NULL;
  for (int i = 0; i < POPULATION_SIZE; i++) {
    g[i].conns = init_conns();
    if (!g[i].conns) {
      for (int j = 0; j < i; j++) {
        DA_FREE(g[j].conns);
        DA_FREE(g[j].nodes);
      }
      free(g);
      return NULL;
    }
    g[i].nodes = init_nodes();
    if (!g[i].nodes) {
      for (int j = 0; j < i; j++) {
        DA_FREE(g[j].conns);
        DA_FREE(g[j].nodes);
      }
      DA_FREE(g[i].conns);
      free(g);
      return NULL;
    }
  }
  return g;
}

struct Population *init_population() {
  struct Population *p = malloc(sizeof(*p));
  if (!p)
    return NULL;
  p->genomes = init_genomes();
  if (!p->genomes) {
    free(p);
    return NULL;
  }
  p->history = init_hashtbl();
  if (!p->history) {
    free(p);
    free(p->genomes);
    return NULL;
  }
  return p;
}

void dump(struct Population *p) {
  for (int i = 0; i < POPULATION_SIZE; i++) {
    printf("Genome %d:\n", i);
    struct Genome *g = &p->genomes[i];
    printf("Nodes: %zu\n", g->nodes->size);
    for (int j = 0; j < g->nodes->size; j++) {
      struct Node *n = &DA_AT(g->nodes, j);
      printf("\t[%2zu] %s | %7.3f\n", n->id,
             n->type == 0   ? "INPUT "
             : n->type == 1 ? "BIAS  "
             : n->type == 2 ? "HIDDEN"
             : n->type == 3 ? "OUTPUT"
                            : "???   ",
             n->value);
    }
    printf("Connections: %zu\n", g->conns->size);
    for (int j = 0; j < g->conns->size; j++) {
      struct Conn *c = &DA_AT(g->conns, j);
      printf("\t[%2zu] %zu %zu | %7.3f\n", c->id, c->in, c->out, c->weight);
    }
  }
}

void dump_dot(struct Population *p, const char *fname) {
  FILE *f = fopen(fname, "w");
  if (!f) {
    fprintf(stderr, "Error opening file %s\n", fname);
    return;
  }

  fprintf(f, "digraph Genome {\n");
  fprintf(f, "\trankdir=LR;\n");

  // For this example, we'll just dump the first genome
  struct Genome *g = &p->genomes[0];

  // Declare nodes with styles
  for (size_t i = 0; i < g->nodes->size; i++) {
    struct Node *n = &DA_AT(g->nodes, i);
    const char *shape = "circle";
    const char *color = "grey";
    if (n->type == INPUT) {
      shape = "box";
      color = "green";
    } else if (n->type == OUTPUT) {
      shape = "box";
      color = "blue";
    }
    fprintf(
        f, "\t%zu [shape=%s, style=filled, fillcolor=%s, label=\"ID: %zu\"];\n",
        n->id, shape, color, n->id);
  }

  // Declare connections
  for (size_t i = 0; i < g->conns->size; i++) {
    struct Conn *c = &DA_AT(g->conns, i);
    const char *style = c->enabled ? "solid" : "dotted";
    const char *color = c->enabled ? "black" : "grey";
    fprintf(f, "\t%zu -> %zu [label=\"%.2f\", style=%s, color=%s];\n", c->in,
            c->out, c->weight, style, color);
  }

  fprintf(f, "}\n");
  fclose(f);
}

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
  for (int i = 0; i < MUT_MAX_ATTEMPTS; i++) {
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

    if (g->conns->size + 1 >= g->conns->cap) {
      g->conns->data =
          // TODO: realloc check
          realloc(g->conns->data, 2 * g->conns->cap * sizeof(*g->conns->data));
      g->conns->cap *= 2;
    }
    struct Conn *c = &DA_AT(g->conns, g->conns->size++);
    c->in = in->id;
    c->out = out->id;
    if (c->in > c->out && out->type != OUTPUT) {
      c->in ^= c->out;
      c->out ^= c->in;
      c->in ^= c->out;
    }
    c->weight = 1.0f;
    c->enabled = TRUE;
    c->id = ht_insert(h, c->in, c->out);
    // TODO: insert check
    // TODO: mutate check
    return;
  }
}

void mut_add_node(struct Genome *g, struct Hashtbl *h) {
  // TODO: history check
  if (frand1() > P_MUT_ADD_NODE)
    return;
  if (g->conns->size == 0)
    return;
  if (g->nodes->size + 1 >= g->nodes->cap) {
    g->nodes->data =
        // TODO: realloc check
        realloc(g->nodes->data, 2 * g->nodes->cap * sizeof(*g->nodes->data));
    g->nodes->cap *= 2;
  }
  struct Node *n = &DA_AT(g->nodes, g->nodes->size++);
  n->id = next_node_id++;
  n->type = HIDDEN;
  n->value = 0.0f;
  struct Conn *c = &DA_AT(g->conns, random() % g->conns->size);
  c->enabled = FALSE;
  if (g->conns->size + 2 >= g->conns->cap) {
    g->conns->data =
        // TODO: realloc check
        realloc(g->conns->data, 2 * g->conns->cap * sizeof(*g->conns->data));
    g->conns->cap *= 2;
  }
  struct Conn *n1 = &DA_AT(g->conns, g->conns->size++);
  n1->in = c->in;
  n1->out = n->id;
  n1->weight = 1.0f;
  n1->enabled = TRUE;
  n1->id = ht_insert(h, n1->in, n1->out);
  // TODO: insert check
  struct Conn *n2 = &DA_AT(g->conns, g->conns->size++);
  n2->in = n->id;
  n2->out = c->out;
  n2->weight = c->weight;
  n2->enabled = TRUE;
  n2->id = ht_insert(h, n2->in, n2->out);
  // TODO: insert check
}

void mutate(struct Population *p) {
  /*
   * There was an 80% chance of a genome having its connection weights mutated,
   * in which case each weight had a 90% chance of being uniformly perturbed and
   * a 10% chance of being assigned a new random value. There was a 75% chance
   * that an inherited gene was disabled if it was disabled in either parent. In
   * each generation, 25% of offspring resulted from mutation without crossover.
   * The interspecies mating rate was 0.001. In smaller populations, the
   * probability of adding a new node was 0.03 and the probability of a new link
   * mutation was 0.05.
   */
  struct Hashtbl *h = p->history;
  for (int i = 0; i < POPULATION_SIZE; i++) {
    struct Genome *g = &p->genomes[i];
    mutate_weights(g);
    mut_add_conn(g, h);
    mut_add_node(g, h);
  }
}

int main(int argc, char *argv[]) {
  // srandom(time(NULL));
  srandom(1);
  struct Population *p = init_population();

  dump(p);

  mutate(p);
  mutate(p);
  mutate(p);

  dump(p);
  // dump_dot(p, "genome_mutated.dot");

  wipe(p);

  // printf("Use 'dot -Tpng genome_mutated.dot -o genome_mutated.png' to
  // render.\n");

  return EXIT_SUCCESS;
}
