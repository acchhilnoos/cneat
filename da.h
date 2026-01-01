#ifndef DA_H
#define DA_H

#define DA_ADD(DA, EL)                                           \
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

#endif
