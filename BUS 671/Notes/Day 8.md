# Day 8

## Normalization

In relational databases a normal form is a set of rules or standards used to organize data in a table to reduce redundancy and improve data integrity

the process of decomposing the column names with anomalies to produce smaller well structured relations

Normalization is a formal process for deciding which attributes should be grouped together in a relation so all anomalies are removed

### Goals of Normalization

Minimizing data redundancy therby avoiding anomalies\

Make it easier to maintain data

Save storage space

### Rules

First

second

third normal

Each form builds on the previous one with stricter requirements

## First normal form

A relation is in the first form if each row has one or more primary keys that distinguishes it as unique

there is no multi-valued attributes all atomic on entry per cell

## Functional dependencies

functional dependencies is a relationship between two attributes in a relation that shows how one attribute depends on another

A constraint between tow attributes in which the value of one attribute is determined by the value of another attribute

## Full partial dependncy

Full dependency

an attribute is fully dependent on all parts of the primary key

Partial dependency

an attribute that is dependent on part of the primary key but not part of the primary key

## Second normal form

It is first normal form and

contains no partioal dependencies

Extract partial dependencies into a new relation
