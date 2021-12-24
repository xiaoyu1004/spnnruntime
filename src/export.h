#ifndef EXPORT_H
#define EXPORT_H

#if defined(_MSC_VER)
#ifdef EXPORT_SPNNRUNTIME
#define SPNN_EXPORT __declspec(dllexport)
#else
#define SPNN_EXPORT __declspec(dllimport)
#endif
#else
#define SPNN_EXPORT __attribute__((visibility("default")))
#endif

#endif