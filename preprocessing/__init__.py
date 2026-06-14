"""Preprocessing package — Document parsing, anonymization, and chunking.

Pipeline order: parse → anonymize → chunk.
Anonymization MUST happen before any agent sees a document.
"""
