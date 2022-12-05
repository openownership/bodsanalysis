import pandas as pd
import re
import statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import inspect
from IPython.display import Markdown as md
import datetime
from datetime import datetime, date
import networkx as nx
import random
import os

cols = ['#363487', '#9bb9d3', '#006a9e', '#f2f4f8', '#dfd5c3', '#f5a00d']


def getBodsData(url):
    """ 
    Description: downloads and unzips BODS data in csv format
    Arguments:
        folder: url link to the csv files from the BODS data analysis tools csv output
        Returns: a folder called 'csv' containing all data, downloaded to the current working directory
    """
    os.system("rm -rf csv* *json  output/*")
    os.system("wget " + url)
    os.system("unzip csv")
    os.system("rm csv.zip")


def readBodsData(folder):
    """ 
    Description: Reads in a set of BODS data in csv format

    Arguments:
        folder: a folder containing ONLY the csv files from the BODS data analysis tools csv output
    Returns: a dictionary of dataframes, with names corresponding to the filename handle
    """
    files = os.listdir(folder)
    output = {}
    for file in files:
        output[file[:-4]] = pd.read_csv('csv/'+file)
    return (output)


def getQueryDescription(query):
    """ 
    Description: Gets the query description from a qbods query

    Arguments:
        Query: a function in the qbods module. This MUST have the query description on a single
        line in the docstring that start with 'Description:'

    Returns: a markdown string containing the query description
    """
    f = inspect.getsource(query)
    flist = f.split('\n')

    for item in flist:
        if item.strip().startswith('Description:'):
            return (md(item.strip()))
            break


def bodsInfo(data):
    """ 
    Description: Summarises a BODS data frame

    Arguments:
        data: a BODS dataframe read in from the BODS data analysis tools csv format    
    Returns: a dataframe containing the the number and proportion of non-missing entries in each column of the input dataframe. This table is also written as tab delimited text to the output folder if an outname is provided (this needs to exist)
    """
    out = data.count(0).to_frame().rename(
        columns={0: 'Number non-missing entries'})
    cond = out.index.str.contains('_link')
    out = out.drop(out[cond].index.values)
    out['Proportion non-missing entries'] = out['Number non-missing entries'] / \
        len(data)
    out = out.round(2)
    return (out)


def propObjectInStatement(obj, statement):
    """ 
    Description: calculates the proportion of statements that contain a given object

    Arguments:
            obj: a BODS dataframe that is an object within an entity statement, person statement
            or ownership or control statement
            statement: a BODS dataframe containing the higher level statements within wich obj 
            is nested

    Returns: a float representing the proportion of statements that contain at least one entry for
            the object
    """
    colnames = list(obj)
    colname = [colname for colname in colnames if '_link_' in colname][0]
    out = len(obj[colname].unique())/len(statement)
    return (out)


def camel_to_snake(name):
    """ 
    Description: Converts a string from camel to snake case

    Arguments:
            name: a string in camel case

    Returns: a string in snake case
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', name).lower()


def read_codelist(url, case='camel'):
    """ 
    Description: reads in a BODS codelist in csv format

    Arguments:
        url: the url containing the raw csv data
        case: a string specifying the case to return the codelist
        must be 'camel' or 'snake'

    Returns: a list of codelist items 
    """
    df = pd.read_csv(url)
    if case == 'snake':  # can return in snake case with required
        return (df['code'].apply(camel_to_snake).tolist())
    else:
        if case == 'camel':
            return (df['code'].tolist())


def sortCounts(df):
    """
    Description: Takes an output from a qbods query and puts columns and rows with 'missing' or 'all' at the end. Also renames 'all' to 'total'

    Arguments:     
        df: a qbods set of counts in pandas dataframe format

    Returns: a resorted and renamed dataframe
    """
    def moveToEnd(l, string):
        l.append(l.pop(l.index(string)))
        return (l)
    if 'Missing' in df.index:
        l = df.index.tolist()
        l = moveToEnd(l, 'Missing')
        df = df.reindex(l)
    if 'Missing' in list(df):
        df.columns = moveToEnd(list(df), 'Missing')
    if 'All' in df.index:
        l = df.index.tolist()
        l = moveToEnd(l, 'All')
        df = df.reindex(l).rename(index={'All': 'Total'})
    if 'All' in list(df):
        df.columns = moveToEnd(list(df), 'All')
        df = df.rename(columns={'All': 'Total'})
    return (df)


def q111(ownershipOrControlInterest, ownershipOrControlStatement):
    """ 
    Description: Provides a breakdown of the beneficial ownership or control field in ownership or control statement interests

    Arguments:
        ownershipOrControlInterests: a pandas dataframe containing ownership or control interests data
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: a list containing a pandas dataframe of counts, and a barplot of the same counts
    """
    # join interests table with ownership or control statements table to get missing values
    df = pd.merge(ownershipOrControlStatement['_link'],
                  ownershipOrControlInterest[[
                      '_link_ooc_statement', 'type', 'beneficialOwnershipOrControl']],
                  how='left',
                  left_on='_link',
                  right_on='_link_ooc_statement')

    # Generate output table
    out = df['beneficialOwnershipOrControl'].fillna(
        'Missing').value_counts().to_frame()
    out = sortCounts(out)

    # Make figure
    ax = out.plot.barh(stacked=True)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()
    return ([out, fig])


def q121(ownershipOrControlInterest, ownershipOrControlStatement, personStatement, entityStatement):
    """ 
    Description: Provides a breakdown of interested parties in ownership or control statements according to whether at least one beneficial ownership or control interest is declared

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests           
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements
        personStatement: a pandas dataframe containing person statements
        entityStatement: a pandas dataframe containing entity statements

    Returns: a list containing a pandas dataframe of counts, and a barplot of the same counts.
    """

    # Merge OOC and person statement tables by interested party
    ooc_ps = pd.merge(ownershipOrControlStatement.dropna(subset=['interestedParty_describedByPersonStatement'])[['_link', 'interestedParty_describedByPersonStatement']],
                      personStatement[['statementID', 'personType']],
                      left_on='interestedParty_describedByPersonStatement',
                      right_on='statementID',
                      how='left')

    # Now merge with interests
    ooci_int = ownershipOrControlInterest[[
        '_link_ooc_statement', 'beneficialOwnershipOrControl']].replace({True: 1, False: 0})
    interestSums = ooci_int.groupby('_link_ooc_statement').agg(
        beneficialOwnershipOrControl=pd.NamedAgg(
            column='beneficialOwnershipOrControl', aggfunc=sum)
    )
    interestSums[interestSums > 1] = 1
    interestSums['beneficialOwnershipOrControl'] = interestSums['beneficialOwnershipOrControl'].replace({
                                                                                                        1: True, 0: False})

    ooc_ps_ooci = pd.merge(ooc_ps,
                           interestSums,
                           left_on='_link',
                           right_on='_link_ooc_statement',
                           how='left')

    # Merge OOC and entity statement tables by interested party
    ooc_es = pd.merge(ownershipOrControlStatement.dropna(subset=['interestedParty_describedByEntityStatement'])[['_link', 'interestedParty_describedByEntityStatement']],
                      entityStatement[['statementID', 'entityType']],
                      left_on='interestedParty_describedByEntityStatement',
                      right_on='statementID',
                      how='left')

    # Now merge with interests
    ooc_es_ooci = pd.merge(ooc_es,
                           interestSums,
                           left_on='_link',
                           right_on='_link_ooc_statement',
                           how='left')

    # Bind person and entity statements
    ooc_ps_es_ooci = ooc_ps_ooci[['personType', 'beneficialOwnershipOrControl']].rename(columns={'personType': 'ownerType'}).append(
        ooc_es_ooci[['entityType', 'beneficialOwnershipOrControl']].rename(
            columns={'entityType': 'ownerType'})
    )

    # Crosstab of person types of with beneficial ownership
    ownerCounts = pd.crosstab(ooc_ps_es_ooci['ownerType'].fillna(
        'Missing'), ooc_ps_es_ooci['beneficialOwnershipOrControl'].fillna('Missing'), margins=True)

    if True not in ownerCounts.columns:
        ownerCounts[True] = 0

    if False not in ownerCounts.columns:
        ownerCounts[False] = 0

    if 'Missing' not in ownerCounts.columns:
        ownerCounts['Missing'] = 0

    out = ownerCounts[[True, False, 'Missing', 'All']]

    # Read in all person and entity types from codelist and append
    ptypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/personType.csv',
                           case='camel')
    etypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/entityType.csv',
                           case='camel')
    alltypes = ptypes+etypes+['All']

    # Add missing codelists and fill with zero
    out = out.reindex(index=alltypes, fill_value=0)
    out = sortCounts(out).rename(columns={
        True: 'BO interests', False: 'Non-BO interests', 'Missing': 'BO data missing'})
    out.index.name = 'Interested party type'
    # Barplot
    ax = out.drop(columns=['Total']).drop(
        index=['Total']).plot.barh(stacked=True, color=cols)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q122(ownershipOrControlInterest, ownershipOrControlStatement, entityStatement):
    """ 
    Description: Provides a count of the number of instances where an entity is listed as a beneficial owner alongside the jurisdiction of declaring companies and interested parties where an entity is listed as a BO

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests            
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements
        entityStatement: a pandas dataframe containing entity statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts. If there are no entities of beneficial owners a list of two placeholder strings is returned with this message
    """

    # Merge interests where BO == True with ooc statements and filter for ownership by entities
    df = pd.merge(ownershipOrControlInterest[ownershipOrControlInterest['beneficialOwnershipOrControl'] == True]['_link_ooc_statement'],
                  ownershipOrControlStatement[[
                      '_link', 'subject_describedByEntityStatement', 'interestedParty_describedByEntityStatement']],
                  how='left',
                  left_on='_link_ooc_statement',
                  right_on='_link')

    df = df.loc[df['interestedParty_describedByEntityStatement'].notnull()]

    # Only do the rest if there are entities with beneificial owners
    if len(df) > 0:
        df = pd.merge(df,
                      entityStatement[['statementID',
                                       'incorporatedInJurisdiction_name']],
                      how='left',
                      left_on='subject_describedByEntityStatement',
                      right_on='statementID')
        df = df.drop_duplicates(subset=['statementID'])
        out = df['incorporatedInJurisdiction_name'].fillna(
            'Missing').value_counts().to_frame()
        # Add totals row
        out = out.append(pd.DataFrame(sum(out['incorporatedInJurisdiction_name']),
                                      columns=[
                                          'incorporatedInJurisdiction_name'],
                                      index=['All']))
        # Plot
        ax = out.drop(labels=['Total']).head(10).plot.barh()
        ax.set(ylabel=None, xlabel="Number of OOC statements")
        fig = ax.get_figure()
        plt.close()
        return ([out, fig])
    else:
        print('No entities with other entities as beneficial owners')
        return (['No table returned', 'No plot returned'])


def q131(ownershipOrControlInterest, ownershipOrControlStatement):
    """ 
    Description: Checks the breakdown of interest types and beneficial ownership or control flags in ownership or control statements

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests            
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """
    # merge ooci table with ooc table to get missing values
    df = pd.merge(ownershipOrControlStatement['_link'],
                  ownershipOrControlInterest[[
                      '_link_ooc_statement', 'type', 'beneficialOwnershipOrControl']],
                  how='left',
                  left_on='_link',
                  right_on='_link_ooc_statement')

    # Generate table with crosstab
    out = pd.crosstab(df['type'].fillna(
        'Missing'), df['beneficialOwnershipOrControl'].fillna('Missing'), margins=True)

    # Add in missing codelist entries and replace with zeros
    rights = read_codelist(
        'https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/interestType.csv')
    out = out.reindex(index=rights+['Missing', 'All'], fill_value=0)

    # Sort
    out = out.sort_values(by=['All'], ascending=False)
    out = sortCounts(out).rename(columns={
        True: 'BO interests', False: 'Non-BO interests', 'Missing': 'BO data missing'})

    # Graph
    ax = out.drop(labels=['Total']).drop(
        columns=['Total']).plot.barh(stacked=True, color=cols)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q132(ownershipOrControlInterest, ownershipOrControlStatement):
    """ 
    Description: Checks the breakdown of direct and indirect interests in ownership or control statements

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests            
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """
    # Generate output table with crosstab
    df = pd.merge(ownershipOrControlStatement['_link'],
                  ownershipOrControlInterest[[
                      '_link_ooc_statement', 'interestLevel', 'beneficialOwnershipOrControl']],
                  how='left',
                  left_on='_link',
                  right_on='_link_ooc_statement')

    out = pd.crosstab(df['interestLevel'].fillna(
        'Missing'), df['beneficialOwnershipOrControl'].fillna('Missing'), margins=True)

    levels = read_codelist(
        'https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/interestLevel.csv')
    out = out.reindex(index=levels+['Missing', 'All'], fill_value=0)
    out = out.sort_values(by='All', ascending=False).rename(
        columns={False: 'False', True: 'True'})
    out = sortCounts(out)

    # Plot
    ax = out.drop(labels=['Total']).drop(
        columns=['Total']).plot.barh(stacked=True)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()
    return ([out, fig])


def q141(ownershipOrControlInterest):
    """ 
   Description: Determines whether the data contain minimum, maximum and/or exact shares.

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests            

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """
    shares = ['share_exact', 'share_minimum', 'share_maximum']
    df = ownershipOrControlInterest.reindex(columns=shares, fill_value=None)
    out = df.notna().sum().to_frame(name='Number non-missing values')
    out = sortCounts(out)

    ax = out.plot.barh(legend=False)
    ax.set(ylabel=None, xlabel='Number of non-missing values')
    fig = ax.get_figure()
    plt.close()
    return ([out, fig])


def q142(ownershipOrControlInterest, threshold):
    """ 
    Description: Determines whether the data contain minimum, maximum and/or exact shares.

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests
        threshold: a number corresponding to the declaration threshold in the declaring country, in percent           
    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """
    shares = ['share_exact', 'share_minimum', 'share_maximum']
    df = ownershipOrControlInterest.reindex(columns=shares, fill_value=None)
    df['max-min share'] = df['share_maximum'] - df['share_minimum']
    idx = ['Most common value', 'Number of unique values',
           'Minimum share', 'Maximum share']
    out = df.mode().append(df.nunique(), ignore_index=True).append(
        df.min(), ignore_index=True).append(df.max(), ignore_index=True)
    out.index = idx

    finalout = [out]

    if out['share_exact']['Number of unique values'] > 0:
        ax = df['share_exact'].hist()
        ax.set(xlabel='Exact share (%)', ylabel='Number of entries')
        ax.axvline(threshold)
        fig1 = ax.get_figure()
        plt.close()
        finalout.append(fig1)

    if out['share_minimum']['Number of unique values'] > 0:
        ax1 = df['share_minimum'].hist()
        ax1.set(xlabel='Minimum share (%)', ylabel='Number of entries')
        ax1.axvline(threshold)
        fig2 = ax1.get_figure()
        plt.close()
        finalout.append(fig2)

    if out['share_maximum']['Number of unique values'] > 0:
        ax2 = df['share_maximum'].hist()
        ax2.set(xlabel='Maximum share (%)', ylabel='Number of entries')
        ax2.axvline(threshold)
        fig3 = ax2.get_figure()
        plt.close()
        finalout.append(fig3)

    return (finalout)


def q211(entityStatement):
    """ 
    Description: Calculates the number of entities that have names

    Arguments:
        entityStatement: a pandas dataframe containing entity statements        

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """
    d = {'Name': [sum(entityStatement['name'].notna()),
                  sum(entityStatement['name'].isna())]}
    out = pd.DataFrame(d, index=['Present', 'Missing'])
    ax = out.plot.barh(legend=False)
    ax.set(xlabel=None, ylabel='Number of entries')
    fig = ax.get_figure()
    plt.close()
    return ([out, fig])


def q212(entityStatement, entityIdentifier):
    """
    Description: Checks the breakdown of entity types across all entity statements, alongside the presence/absence of identifiers, to determine which types of entities provide identifying information.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        entityIdentifier: a pandas dataframe containing entity identifiers

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """

    # Add a column determining if identifiers present and do a crosstab
    es_subset = entityStatement.assign(identifiersPresent=entityStatement['_link'].isin(
        entityIdentifier['_link_entity_statement']).tolist())[['identifiersPresent', 'entityType']]
    out = pd.crosstab(es_subset['entityType'].fillna(
        'Missing'), es_subset['identifiersPresent'].fillna('Missing'), margins=True)

    if True not in out.columns:
        out[True] = 0

    if False not in out.columns:
        out[False] = 0

    if 'Missing' not in out.columns:
        out['Missing'] = 0

    out = out[[True, False, 'Missing', 'All']]

    # Read in all person and entity types from codelist and append
    etypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/entityType.csv',
                           case='camel')
    alltypes = etypes+['All']

    # Add missing codelists and fill with zero
    out = out.reindex(index=alltypes, fill_value=0).rename(
        columns={True: 'True', False: 'False'})
    out = sortCounts(out)

    # Plot
    ax = out.drop(labels=['Total']).drop(
        columns=['Total']).plot.barh(stacked=True)
    ax.set(ylabel=None, xlabel="Number of entity statements")
    fig = ax.get_figure()
    plt.close()
    return ([out, fig])


def q213(ownershipOrControlInterest, ownershipOrControlStatement, entityStatement):
    """ 
    Description: Provides a breakdown of subjects in ownership or control statements according to whether at least one beneficial ownership or control interest is declared

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests           
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements
        entityStatement: a pandas dataframe containing entity statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts.
    """

    # Merge OOC and entity statement tables by subject
    ooc_es = pd.merge(ownershipOrControlStatement.dropna(subset=['subject_describedByEntityStatement'])[['_link', 'subject_describedByEntityStatement']],
                      entityStatement[['statementID', 'entityType']],
                      left_on='subject_describedByEntityStatement',
                      right_on='statementID',
                      how='left')

    # Now merge with interests
    ooci_int = ownershipOrControlInterest[[
        '_link_ooc_statement', 'beneficialOwnershipOrControl']].replace({True: 1, False: 0})
    interestSums = ooci_int.groupby('_link_ooc_statement').agg(
        beneficialOwnershipOrControl=pd.NamedAgg(
            column='beneficialOwnershipOrControl', aggfunc=sum)
    )
    interestSums[interestSums > 1] = 1
    interestSums['beneficialOwnershipOrControl'] = interestSums['beneficialOwnershipOrControl'].replace({
                                                                                                        1: True, 0: False})

    ooc_es_ooci = pd.merge(ooc_es,
                           interestSums,
                           left_on='_link',
                           right_on='_link_ooc_statement',
                           how='left')

    # Crosstab of person types of with beneficial ownership
    ownerCounts = pd.crosstab(ooc_es_ooci['entityType'].fillna(
        'Missing'), ooc_es_ooci['beneficialOwnershipOrControl'].fillna('Missing'), margins=True)

    if True not in ownerCounts.columns:
        ownerCounts[True] = 0

    if False not in ownerCounts.columns:
        ownerCounts[False] = 0

    if 'Missing' not in ownerCounts.columns:
        ownerCounts['Missing'] = 0

    out = ownerCounts[[True, False, 'Missing', 'All']]

    # Read in all person and entity types from codelist and append
    etypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/entityType.csv',
                           case='camel')
    alltypes = etypes+['All']

    # Add missing codelists and fill with zero
    out = out.reindex(index=alltypes, fill_value=0)
    out = sortCounts(out).rename(columns={
        True: 'BO interests', False: 'Non-BO interests', 'Missing': 'BO data missing'})

    # Barplot
    ax = out.drop(columns=['Total'], index=['Total']
                  ).plot.barh(stacked=True, color=cols)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q214(ownershipOrControlStatement, entityStatement):
    """
    Description: Provides a breakdown of entities by jurisdiction and counts the number of entities that do not list a jurisdiction. This is further broken down into whether entities are subjects or interested parties in ownership or control statements.

    Arguments:     
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements
        entityStatement: a pandas dataframe containing entity statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts, with countries with fewer than 500 statements grouped together. 
    """
    # Get single column of entities with column idneitfying if they're subjects or interested parties
    ooc_long = ownershipOrControlStatement.assign(type='Subject')[['subject_describedByEntityStatement', 'type']].rename(columns={'subject_describedByEntityStatement': 'statementID'}).append(
        ownershipOrControlStatement.assign(type='Interested party')[['interestedParty_describedByEntityStatement', 'type']].rename(columns={'interestedParty_describedByEntityStatement': 'statementID'})).dropna()

    # #Join to entity statemennts
    ooc_es_long = pd.merge(ooc_long,
                           entityStatement[['statementID',
                                            'incorporatedInJurisdiction_name']],
                           left_on='statementID',
                           right_on='statementID',
                           how='left')

    out = pd.crosstab(ooc_es_long['incorporatedInJurisdiction_name'].fillna('Missing'),
                      ooc_es_long['type'].fillna('Missing'), margins=True)

    out = sortCounts(out)

    # Barplot - group countries with less than 500 ooc statements together
    outplot = out.copy(deep=True)
    outplot['jurisdictionOther'] = np.where(
        outplot['Total'] > 500, outplot.index, 'Other')
    outplot = outplot.groupby(['jurisdictionOther']).sum(
    ).sort_values(by=['Total'], ascending=False)
    ax = outplot.drop(columns=['Total'], index=[
                      'Total']).plot.barh(stacked=True)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q221(entityStatement):
    """
    Description: Provides a count of reasons given for not displaying entity details, broken down by entity type, alongside the number of descriptions provided

    Arguments:     
        entityStatement: a pandas dataframe containing entity statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts
    """

    unspecReason = pd.crosstab(entityStatement['unspecifiedEntityDetails_reason'].fillna('Missing'),
                               entityStatement['entityType'].fillna('Missing'),
                               margins=True)

    unspecDescription = entityStatement.copy(deep=True)
    if 'unspecifiedEntityDetails_description' in list(entityStatement):
        unspecDescription['descriptionProvided'] = unspecDescription['unspecifiedEntityDetails_description'].notna()
    else:
        unspecDescription['descriptionProvided'] = False

    unspecDescription = unspecDescription[[
        'unspecifiedEntityDetails_reason', 'descriptionProvided']]
    unspecDescription = unspecDescription.groupby(
        ['unspecifiedEntityDetails_reason']).sum()

    out = pd.concat([unspecReason, unspecDescription], axis=1).fillna(int(0))

    # Read in all entity types from codelist and append
    reasontypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/unspecifiedReason.csv',
                                case='camel')+['Missing', 'All']
    etypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/entityType.csv',
                           case='camel')+['Missing', 'All']

    # Add missing codelists and fill with zero
    out = out.reindex(index=reasontypes, columns=etypes +
                      ['descriptionProvided'], fill_value=0)
    out = sortCounts(out)

    # Barplot
    ax = out.drop(columns=['Total', 'Missing', 'descriptionProvided'], index=[
                  'Total', 'Missing']).plot.barh(stacked=True, color=cols)
    ax.set(ylabel=None, xlabel="Number of OOC statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q222(personStatement):
    """
    Description: Provides a count of reasons given for not displaying person details, broken down by person type, alongside the number of descriptions provided

    Arguments:     
        personStatement: a pandas dataframe containing person statements

    Returns: a list containing i) a pandas dataframe of counts, and ii) a barplot of the same counts 
    """
    unspecReason = pd.crosstab(personStatement['unspecifiedPersonDetails_reason'].fillna('Missing'),
                               personStatement['personType'].fillna('Missing'),
                               margins=True)

    unspecDescription = personStatement.copy(deep=True)

    if 'unspecifiedPersonDetails_description' in list(personStatement):
        unspecDescription['descriptionProvided'] = unspecDescription['unspecifiedPersonDetails_description'].notna()
    else:
        unspecDescription['descriptionProvided'] = False

    unspecDescription = unspecDescription[[
        'unspecifiedPersonDetails_reason', 'descriptionProvided']]
    unspecDescription = unspecDescription.groupby(
        ['unspecifiedPersonDetails_reason']).sum()

    out = pd.concat([unspecReason, unspecDescription], axis=1).fillna(int(0))

    # Read in all person types from codelist and append
    reasontypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/unspecifiedReason.csv',
                                case='camel')+['Missing', 'All']
    ptypes = read_codelist('https://raw.githubusercontent.com/openownership/data-standard/0.2.0/schema/codelists/personType.csv',
                           case='camel')+['Missing', 'All']

    # Add missing codelists and fill with zero
    out = out.reindex(index=reasontypes, columns=ptypes +
                      ['descriptionProvided'], fill_value=0)
    out = sortCounts(out)

    # Barplot
    ax = out.drop(columns=['Total', 'descriptionProvided'], index=[
                  'Total', 'Missing']).plot.barh(stacked=True)
    ax.set(ylabel=None, xlabel="Number of person statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q223(entityStatement, personStatement):
    """
    Description: Provides a set of unique descriptions for unspecified entities or persons:

    Arguments:
        entityStatement: a pandas dataframe containing entity statements     
        personStatement: a pandas dataframe containing person statements

    Returns: a string with one line per unique description
    """
    esout = []
    psout = []

    if 'unspecifiedEntityDetails_description' in list(entityStatement):
        esout = esout + entityStatement[entityStatement['unspecifiedEntityDetails_description'].notna(
        )]['unspecifiedEntityDetails_description'].unique().tolist()
        esout = ['unspecified entity description: ' + item for item in esout]
    if 'unspecifiedPersonDetails_description' in list(personStatement):
        psout = psout + personStatement[personStatement['unspecifiedPersonDetails_description'].notna(
        )]['unspecifiedPersonDetails_description'].unique().tolist()
        psout = ['unspecified person description: ' + item for item in psout]

    out = '\n'.join([str(len(esout)+len(psout)) +
                    ' statements detected ---']+esout+psout)
    return (print(out))


def q224(personStatement, minimumAge, maximumAge):
    """
    Description: If age-based exemptions from declaration exist, provides a count of person statements are within and outside of the eligible age range.

    Arguments:
        personStatement: a pandas dataframe containing person statements
        minimumAge: The youngest age for which declaration is permitted under reporting regime
        maximumAge: The oldest age for which declaration is permitted under reporting regime

    Returns: A list containig a dataframe of counts, and corresponding barplot of the same counts
    """
    now = pd.Timestamp('now')
    psdate = personStatement.copy(deep=True)
    psdate['birthDate'] = pd.to_datetime(psdate['birthDate'])
    psdate['birthDate'] = psdate['birthDate'].where(
        psdate['birthDate'] < now, psdate['birthDate'] - np.timedelta64(100, 'Y'))
    psdate['age'] = (now - psdate['birthDate']).astype('<m8[Y]')

    d = {'Count': [sum((psdate['age'] >= minimumAge) & (psdate['age'] <= maximumAge)),
                   sum((psdate['age'] < minimumAge) |
                       (psdate['age'] > maximumAge)),
                   sum(psdate['age'].isna()),
                   len(psdate)]}
    out = pd.DataFrame(data=d,
                       index=['Within eligible range', 'Outside eligible range', 'Missing', 'All'])

    out = sortCounts(out)

    # Barplot
    ax = out.drop(index=['Total']).plot.barh(stacked=True)
    ax.set(ylabel=None, xlabel="Number of person statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q311(personStatement, personNationalities, personIdentifiers, personNames, declaringCountry):
    """
    Description: Provides a breakdown of presence/absence of birthdates, identifiers, names, for domestic and foreign nationals

    Arguments:
        personStatement: a pandas dataframe containing person statements
        personNationalities: a pandas dataframe containing person nationalities
        personIdenifiers: a pandas dataframe containing person identifiers
        personNames: a pandas dataframe containing person names
        declaringCountry: the two letter country code of the declaring country

    Returns: A dataframe of counts, with comma separated values corresponding to number and proportion of non missing values
    """

    ps_pnat = pd.merge(personStatement[['_link', 'birthDate']],
                       personNationalities[['_link_person_statement', 'code']],
                       left_on='_link',
                       right_on='_link_person_statement',
                       how='left').dropna()
    ps_pnat[ps_pnat['code'] != declaringCountry] = 'other'
    ps_pnat['birthDate'] = ps_pnat['birthDate'].notna()
    ps_pnat['identifiers'] = ps_pnat['_link'].isin(
        personIdentifiers['_link_person_statement'])
    ps_pnat['name'] = ps_pnat['_link'].isin(
        personNames['_link_person_statement'])
    out = ps_pnat[['code', 'birthDate', 'identifiers', 'name']
                  ].groupby(['code']).sum().transpose()
    meanout = ps_pnat[['code', 'birthDate', 'identifiers', 'name']].groupby(
        ['code']).mean().transpose().round(2)
    out[declaringCountry] = out[declaringCountry].astype(
        str).str.cat(meanout[declaringCountry].astype(str), sep=', ')
    out['other'] = out['other'].astype(str).str.cat(
        meanout['other'].astype(str), sep=', ')
    out = sortCounts(out)

    return (out)


def q312(personIdentifiers):
    """
    Description: Provides a breakdown of identifiers

    Arguments:
        PersonIdenifiers: a pandas dataframe containing person identifiers

    Returns: A list contaning a dataframe of counts, and a corresponding barplot
    """

    out = personIdentifiers['scheme'].fillna(
        'Missing').value_counts().to_frame()

    out = sortCounts(out)

    # Barplot
    ax = out.plot.barh(legend=False)
    ax.set(ylabel=None, xlabel="Number of identifier statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q313(personNationalities):
    """
    Description: Provides a breakdown of person nationalities

    Arguments
        personNationalities: a pandas dataframe containing person identifiers

    Returns: A list containing a dataframe of counts, and a corresponding barplot, with countries with less than 500 entries grouped as other
    """

    out = personNationalities['name'].fillna(
        'Missing').value_counts().to_frame()

    out = sortCounts(out)

    # Barplot
    outplot = out.copy(deep=True)
    outplot['nameOther'] = np.where(
        outplot['name'] > 500, outplot.index, 'Other')
    outplot = outplot.groupby(['nameOther']).sum(
    ).sort_values(by=['name'], ascending=False)
    ax = outplot.plot.barh(legend=False)
    ax.set(ylabel=None, xlabel="Number of nationality statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q314(personStatement):
    """
    Description: Provides a breakdown of person birth years

    Arguments:
        personStatement: a pandas dataframe containing person statements

    Returns: A dataframe of counts by decade, and a corresponding histogram showing distribution by birth year
    """

    out = personStatement.copy(deep=True)['birthDate'].to_frame()
    out['birthYear'] = pd.DatetimeIndex(out['birthDate']).year/10
    out['birthDecadeStart'] = out['birthYear'].apply(np.floor)*10+1
    out['birthDecadeEnd'] = out['birthDecadeStart']+9
    out['birthYear'] = out['birthYear']*10
    out = out[['birthDecadeStart', 'birthDecadeEnd',
               'birthYear']].dropna().astype(int)
    out['birthDecade'] = out['birthDecadeStart'].astype(
        str).str.cat(out['birthDecadeEnd'].astype(str), sep='-')
    finalout = out['birthDecade'].value_counts(
        sort=False).to_frame().sort_index()
    # Barplot
    ax = out['birthYear'].hist(color=cols[0])
    ax.set(ylabel="Frequency", xlabel="Birth year")
    fig = ax.get_figure()
    plt.close()

    return ([finalout, fig])


def q315(personTaxResidencies):
    """
    Description: Provides a breakdown of tax residencies NOT CHECKED

    Arguments:
        personTaxResidencies: a pandas dataframe containing person tax residencies

    Returns: A list containing a dataframe of counts by tax residency, and a corresponding barplot of the same counts
    """
    out = personTaxResidencies['name'].fillna(
        'Missing').value_counts().to_frame()
    out = sortCounts(out)

    # Barplot
    ax = out.plot.barh(legend=False)
    ax.set(ylabel=None, xlabel="Number of identifier statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q321(entityIdentifiers):
    """
    Description: Provides a breakdown of entity identifiers

    Arguments:
        entityIdentifiers: a pandas dataframe containing entity identifiers

    Returns: A list contaning a dataframe of counts, and a corresponding barplot
    """

    out = entityIdentifiers['scheme'].fillna(
        'Missing').value_counts().to_frame()
    out = sortCounts(out)

    # Barplot
    outplot = out.copy(deep=True)
    outplot['schemeOther'] = np.where(
        outplot['scheme'] > 500, outplot.index, 'Other')
    outplot = outplot.groupby(['schemeOther']).sum(
    ).sort_values(by=['scheme'], ascending=False)

    ax = outplot.plot.barh(legend=False)
    ax.set(ylabel=None, xlabel="Number of identifier statements")
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])

def q322(entityAddresses):
    """
    Description: Provides a breakdown of entity addresses

    Arguments:
        entityAddresses: a pandas dataframe containing entity addresses

    Returns: A list contaning a dataframe of counts
    """

    out = entityAddresses['address'].fillna(
        'Missing').value_counts().to_frame()
    out = sortCounts(out)[0:10]
    return(out)

def q331(ownershipOrControlStatement, samplesize=100):
    """
    Description: Provides a summary of beneficial ownership chains

    Arguments:
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements
        samplesize: how many ownership chains to summarise. Default is 100, and more will be slower

    Returns: A list contaning a dataframe of ownership chain summaries, and a network diagram of ownership chains
    """

    if 'nterestedParty_describedByEntityStatement' in ownershipOrControlStatement:
        subjectList = ownershipOrControlStatement['subject_describedByEntityStatement'].tolist(
        )
        ipPersonList = ownershipOrControlStatement['interestedParty_describedByPersonStatement'].tolist(
        )
        ipEntityList = ownershipOrControlStatement['interestedParty_describedByEntityStatement'].tolist(
        )
        subject = subjectList*2
        interestedParty = ipPersonList+ipEntityList
    else:
        subject = ownershipOrControlStatement['subject_describedByEntityStatement'].tolist(
        )
        interestedParty = ownershipOrControlStatement['interestedParty_describedByPersonStatement'].tolist(
        )


    el = pd.DataFrame(
        {'subject': subject, 'interestedParty': interestedParty}).dropna()

    G = nx.from_pandas_edgelist(el, 'subject', 'interestedParty')
    components = list(nx.connected_components(G))
    idx = random.sample(range(len(components)), samplesize)
    components = [components[i] for i in idx]
    newnodes = list(
        set([node for component in components for node in component]))
    G = G.subgraph(newnodes)

    numberNodes = []
    numberEntities = []
    numberPersons = []

    for component in components:
        numberNodes.append(len(component))
        numberEntities.append(
            sum([node in subjectList+ipEntityList for node in component]))
        numberPersons.append(sum([node in ipPersonList for node in component]))

    out = pd.DataFrame({'numberNodes': numberNodes, 'numberEntities': numberEntities,
                       'numberPersons': numberPersons}).describe()
    out = out.loc[['mean', 'min', 'max']]

    fig = plt.figure()
    ax = nx.draw_networkx(G, pos=nx.spring_layout(
        G), node_size=6, width=0.1, with_labels=False, node_color=cols[0])
    plt.close()

    return ([out, fig])


def q332(ownershipOrControlInterests, ownershipOrControlStatement):
    """
    Description: Checks the completeness of beneficial ownership chains in which indirect interests are described

    Arguments:
        ownershipOrControlInterests: a pandas dataframe containing ownership or control interests
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table describing the completeness of beneficial ownership, and a corresponding barplot
    """

    # indirect and BO OOC statements
    df_indirect = (ownershipOrControlStatement.set_index('_link')
                   .reindex(ownershipOrControlInterests[(ownershipOrControlInterests['interestLevel'] == 'indirect') &
                                                        (ownershipOrControlInterests['beneficialOwnershipOrControl'] == True)]
                            ['_link_ooc_statement']
                            .tolist()))

    # direct ownership statements
    df_direct = (ownershipOrControlStatement.set_index('_link')
                 .reindex(ownershipOrControlInterests[ownershipOrControlInterests['interestLevel'] == 'direct']
                          ['_link_ooc_statement']
                          .tolist()))

    sub_ds = []  # Number direct statements each subject found in
    sub_ds_comp = []  # Number direct statements involving component entity each subject found in

    # Loop over indirect ownership statements
    for i in range(len(df_indirect.index)):
        # Get subject ID and look for it as a subject in the table of direct statements
        if not pd.isnull(df_indirect['subject_describedByEntityStatement'].iloc[i]):
            sub = df_indirect['subject_describedByEntityStatement'].iloc[i]
            sub_data = df_direct[df_direct['subject_describedByEntityStatement'] == sub]

        # nrows of the resulting datasets should be the number of direct statements involving each indirect subject/ip
        sub_ds.append(len(sub_data.index))
        # if resulting data exist, then check how often direct relationship involve a component entity statement
        sub_tempout = 0
        if len(sub_data.index) > 0:
            components = df_indirect['componentStatementIDs'].tolist()[
                i].split(',')
            templist = sub_data['subject_describedByEntityStatement']
            for item in templist:
                if item in components:
                    sub_tempout = sub_tempout+1
        sub_ds_comp.append(sub_tempout)

    # Final output table ---------------
    idx = ['With at least one component entity',
           'With subject found in at least one direct OOC statement',
           'With subject found in at least one direct OOC statement with a corresponding component entity']
    sub_out = [sum(df_indirect['componentStatementIDs'].notna())/len(df_indirect.index)*100,
               sum(sub_ds)/len(df_indirect.index)*100,
               sum(sub_ds_comp)/len(df_indirect.index)*100]

    out = pd.DataFrame(sub_out, index=idx, columns=[
                       '% indirect OOC statements'])

    # Plot -----------------------------
    ax = out.plot.barh(legend=False)

    # Sort out long labels
    labs = ['\n'.join(textwrap.wrap(l, 20)) for l in idx]
    ax.set_yticklabels(labs)
    ax.set(xlabel='% indirect OOC statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q333(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks the number of person statements, entity statements, and ownership or control statements that are explicitly listed as intermediaries in a wider chain of ownership.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the isComponent field in entity, person and OOC statements, and a corresponding barplot
    """

    entityComponent = entityStatement['isComponent'].copy(deep=True).to_frame()
    entityComponent['statementType'] = 'entityStatement'

    personComponent = personStatement['isComponent'].copy(deep=True).to_frame()
    personComponent['statementType'] = 'personStatement'

    oocComponent = ownershipOrControlStatement['isComponent'].copy(
        deep=True).to_frame()
    oocComponent['statementType'] = 'ownershipOrControlStatement'

    allComponent = entityComponent.append(personComponent).append(oocComponent)

    out = pd.crosstab(allComponent['statementType'].fillna('Missing'), allComponent['isComponent'].fillna(
        'Missing'), margins=True).rename(columns={True: 'True', False: 'False'})
    out = sortCounts(out)

    ax = out.drop(index='All', columns='All').plot.barh(stacked=True)

    # Sort out long labels
    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q334(ownershipOrControlInterest):
    """
    Description: Provides a breakdown of the number of Ownership or Control Statements by Interest Start date, and by whether that interest is listed as Beneficial Ownership or Control

    Arguments:
        ownershipOrControlStatement: a pandas dataframe containing ownership or control interests

    Returns: A list containing a summary table of counts by year, and a corresponding barplot
    """

    if 'startDate' in list(ownershipOrControlInterest):
        out = ownershipOrControlInterest.copy(
            deep=True)[['startDate', 'beneficialOwnershipOrControl']].to_frame()
        out['startYear'] = pd.DatetimeIndex(out['startDate']).year
        out = pd.crosstab(out['startYear'].fillna(
            'Missing'), out['beneficialOwnershipOrControl'].fillna('Missing'), margins=True)
        out = sortCounts(out)

        ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)

        # Sort out long labels
        ax.set(xlabel='Number of statements', ylabel=None)

        fig = ax.get_figure()
        plt.close()
        return ([out, fig])
    else:
        print('No interest start dates found')


def q411(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks publisher name of person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the publisher name field in entity, person and OOC statements, and a corresponding barplot
    """

    entityPub = entityStatement['publicationDetails_publisher_name'].copy(
        deep=True).to_frame()
    entityPub['statementType'] = 'entityStatement'

    personPub = personStatement['publicationDetails_publisher_name'].copy(
        deep=True).to_frame()
    personPub['statementType'] = 'personStatement'

    oocPub = ownershipOrControlStatement['publicationDetails_publisher_name'].copy(
        deep=True).to_frame()
    oocPub['statementType'] = 'ownershipOrControlStatement'

    allPub = entityPub.append(personPub).append(oocPub)
    allPub = allPub.rename(
        columns={'publicationDetails_publisher_name': 'publisherName'})

    out = pd.crosstab(allPub['publisherName'].fillna(
        'Missing'), allPub['statementType'].fillna('Missing'), margins=True)

    if 'Missing' not in out.index:
        out = out.reindex(index=out.index.tolist()+['Missing'], fill_value=0)

    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)
    # Sort out long labels
    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q412(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks publisher date of person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the publisher date field in entity, person and OOC statements, and a corresponding barplot. Grouped by year
    """

    entityPub = entityStatement['publicationDetails_publicationDate'].copy(
        deep=True).to_frame()
    entityPub['statementType'] = 'entityStatement'

    personPub = personStatement['publicationDetails_publicationDate'].copy(
        deep=True).to_frame()
    personPub['statementType'] = 'personStatement'

    oocPub = ownershipOrControlStatement['publicationDetails_publicationDate'].copy(
        deep=True).to_frame()
    oocPub['statementType'] = 'ownershipOrControlStatement'

    allPub = entityPub.append(personPub).append(oocPub)
    allPub = allPub.rename(
        columns={'publicationDetails_publicationDate': 'publicationDate'})
    allPub['publicationYear'] = pd.DatetimeIndex(
        allPub['publicationDate']).year

    out = pd.crosstab(allPub['publicationYear'].fillna(
        'Missing'), allPub['statementType'].fillna('Missing'), margins=True)
    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)

    # Sort out long labels
    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q511(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks license details of person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the license field in entity, person and OOC statements, and a corresponding barplot
    """

    entityPub = entityStatement['publicationDetails_license'].copy(
        deep=True).to_frame()
    entityPub['statementType'] = 'entityStatement'

    personPub = personStatement['publicationDetails_license'].copy(
        deep=True).to_frame()
    personPub['statementType'] = 'personStatement'

    oocPub = ownershipOrControlStatement['publicationDetails_license'].copy(
        deep=True).to_frame()
    oocPub['statementType'] = 'ownershipOrControlStatement'

    allPub = entityPub.append(personPub).append(oocPub)
    allPub = allPub.rename(columns={'publicationDetails_license': 'license'})

    out = pd.crosstab(allPub['license'].fillna(
        'Missing'), allPub['statementType'].fillna('Missing'), margins=True)

    if 'Missing' not in out.index:
        out = out.reindex(index=out.index.tolist()+['Missing'], fill_value=0)

    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)

    # Sort out long labels
    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q611(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks BODS version of person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the BODS version field in entity, person and OOC statements, and a corresponding barplot
    """

    entityPub = entityStatement['publicationDetails_bodsVersion'].copy(
        deep=True).to_frame()
    entityPub['statementType'] = 'entityStatement'

    personPub = personStatement['publicationDetails_bodsVersion'].copy(
        deep=True).to_frame()
    personPub['statementType'] = 'personStatement'

    oocPub = ownershipOrControlStatement['publicationDetails_bodsVersion'].copy(
        deep=True).to_frame()
    oocPub['statementType'] = 'ownershipOrControlStatement'

    allPub = entityPub.append(personPub).append(oocPub)
    allPub = allPub.rename(
        columns={'publicationDetails_bodsVersion': 'bodsVersion'})

    out = pd.crosstab(allPub['bodsVersion'].fillna(
        'Missing'), allPub['statementType'].fillna('Missing'), margins=True)

    if 'Missing' not in out.index:
        out = out.reindex(index=out.index.tolist()+['Missing'], fill_value=0)

    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)

    # Sort out long labels
    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q711(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks source type of person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing person statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the source type field in entity, person and OOC statements, and a corresponding barplot
    """

    def splitSource(df):
        dfSource = df['source_type'].copy(deep=True).to_frame()
        dfSource['source_type'] = dfSource['source_type'].str.split(',')
        sourceOut = dfSource.explode('source_type').reset_index(drop=True)
        return (sourceOut)

    entitySource = splitSource(entityStatement)
    entitySource['statementType'] = 'entityStatement'

    personSource = splitSource(personStatement)
    personSource['statementType'] = 'personStatement'

    oocSource = splitSource(ownershipOrControlStatement)
    oocSource['statementType'] = 'ownershipOrControlStatement'

    allSource = entitySource.append(personSource).append(oocSource)

    out = pd.crosstab(allSource['source_type'].fillna(
        'Missing'), allSource['statementType'].fillna('Missing'), margins=True)

    if 'Missing' not in out.index:
        out = out.reindex(index=out.index.tolist()+['Missing'], fill_value=0)

    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)

    # Sort out long labels
    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q811(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks statement date of person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of the statement date field in entity, person and OOC statements, and a corresponding barplot. Grouped by year
    """

    entitySD = entityStatement['statementDate'].copy(deep=True).to_frame()
    entitySD['statementType'] = 'entityStatement'

    personSD = personStatement['statementDate'].copy(deep=True).to_frame()
    personSD['statementType'] = 'personStatement'

    oocSD = ownershipOrControlStatement['statementDate'].copy(
        deep=True).to_frame()
    oocSD['statementType'] = 'ownershipOrControlStatement'

    allSD = entitySD.append(personSD).append(oocSD)
    allSD['statementYear'] = pd.DatetimeIndex(
        allSD['statementDate']).year.astype('Int64')

    out = pd.crosstab(allSD['statementYear'].fillna(
        'Missing'), allSD['statementType'].fillna('Missing'), margins=True)
    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(
        stacked=True, color=cols)

    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q812(ownershipOrControlInterest, ownershipOrControlStatement):
    """
    Description:  Provides a breakdown of the difference between a statement date of ownership or control statements, and the interest start date in that statement

    Arguments:
        ownershipOrControlInterest: a pandas dataframe containing ownership or control interests
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a table of summary statistics of the difference between statement dates and interest start dates, in days, and a corresponding barplot
    """
    oocMerged = pd.merge(ownershipOrControlInterest[['_link_ooc_statement', 'startDate']],
                         ownershipOrControlStatement[[
                             '_link', 'statementDate']],
                         left_on='_link_ooc_statement',
                         right_on='_link',
                         how='left')

    oocMerged['startDate'] = pd.to_datetime(oocMerged['startDate'])
    oocMerged['statementDate'] = pd.to_datetime(oocMerged['statementDate'])
    oocMerged['statementDate-startDate'] = oocMerged['statementDate'] - \
        oocMerged['startDate']
    oocMerged['statementDate-startDate'] = oocMerged['statementDate-startDate'] / \
        np.timedelta64(1, 'D')

    out = oocMerged['statementDate-startDate'].to_frame().describe().drop(index=['std'])

    return (out)


def q821(entityStatement):
    """
    Description:  Provides a breakdown of company founding dates and dissolution dates in entity statements

    Arguments:
        entityStatement: a pandas dataframe containing entity statements

    Returns: A list containing a table of counts of founding dates and dissolution dates grouped into years, and a corresponding figure
    """

    if 'foundingDate' in list(entityStatement):
        fd = entityStatement.copy(deep=True)['foundingDate'].to_frame()
        fd['foundingYear'] = pd.DatetimeIndex(
            fd['foundingDate']).year.astype('Int64')
        fd = fd['foundingYear'].fillna('Missing').value_counts().to_frame()

    if 'dissolutionDate' in list(entityStatement):
        dd = entityStatement.copy(deep=True)['dissolutionDate'].to_frame()
        dd['dissolutionYear'] = pd.DatetimeIndex(
            dd['dissolutionDate']).year.astype('Int64')
        dd = dd['dissolutionYear'].fillna('Missing').value_counts().to_frame()

    if ('foundingDate' in list(entityStatement)) & ('dissolutionDate' in list(entityStatement)):
        out = pd.concat([fd, dd], axis=1)
    else:
        if 'foundingDate' in list(entityStatement):
            out = fd
        else:
            out = dd
    out = sortCounts(out)

    ax = out.plot.barh(stacked=True)
    ax.set(xlabel='Number of entity statements', ylabel=None)
    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q831(entityStatement, personStatement, ownershipOrControlStatement):
    """
    Description: Checks number of non-missing replacesStatements entries in person statements, entity statements, and ownership or control statements.

    Arguments:
        entityStatement: a pandas dataframe containing entity statements
        personStatement: a pandas dataframe containing entity statements
        ownershipOrControlStatement: a pandas dataframe containing ownership or control statements

    Returns: A list containing a summary table of counts of number of non-missing replacesStatements in entity, person and OOC statements, and a corresponding barplot
    """

    def countRS(df):
        dfRS = df.copy(deep=True)
        if 'replacesStatements' not in list(df):
            dfRS['replacesStatements'] = np.nan

        dfRS = dfRS['replacesStatements'].to_frame()
        return (dfRS)

    entityRS = countRS(entityStatement)
    entityRS['statementType'] = 'entityStatement'

    personRS = countRS(personStatement)
    personRS['statementType'] = 'personStatement'

    oocRS = countRS(ownershipOrControlStatement)
    oocRS['statementType'] = 'ownershipOrControlStatement'

    allRS = entityRS.append(personRS).append(oocRS)
    allRS['replacesStatements'] = allRS['replacesStatements'].notna().replace(
        [True, False], ['Present', 'Missing'])

    out = pd.crosstab(allRS['replacesStatements'],
                      allRS['statementType'], margins=True)
    out = sortCounts(out)

    ax = out.drop(index='Total', columns='Total').plot.barh(stacked=True)

    ax.set(xlabel='Number of statements', ylabel=None)

    fig = ax.get_figure()
    plt.close()

    return ([out, fig])


def q832(dict):
    """
    Description: Provides a summary of all date fields in the dataset.

    Arguments:
        dict: the dictionary containing pandas dataframes generated using qbods.readBodsData()

    Returns: A pandas dataframe with all date columns, the number of entries, and some example entries
    """
    nams = []
    nEntries = []
    example1 = []
    example2 = []
    example3 = []

    for item in list(dict.keys()):
        df = dict[item]
        cols = list(df)
        dateCols = [col for col in cols if (
            'Date' in col) | ('retrievedAt' in col)]
        for col in dateCols:
            if col not in nams:
                nams.append(col)
                nEntries.append(sum(df[col].notna()))
                example1.append(df[col].tolist()[0])
                example2.append(df[col].tolist()[1])
                example3.append(df[col].tolist()[2])

            else:
                nEntries[nams == col] = nEntries[nams == col] + \
                    sum(df[col].notna())
    d_out = {'Date column': nams, 'Number entries': nEntries,
             'Example 1': example1, 'Example 2': example2, 'Example 3': example3}
    out = pd.DataFrame(d_out)
    return (out)


def q921(ownershipOrControlStatement):
    """
    Description: Provides a breakdown of reasons for interested parties being unspecified

    Arguments:
        ownershipOrControlStatement: A pandas dataframe of ownership or control statements

    Returns: A pandas dataframe of counts of unspecified reasons
    """
    out = ownershipOrControlStatement['interestedParty_unspecified_reason'].value_counts(
    ).to_frame()
    d = {'interestedParty_unspecified_reason': sum(ownershipOrControlStatement['interestedParty_describedByEntityStatement'].isna(
    ) & ownershipOrControlStatement['interestedParty_describedByPersonStatement'].isna() & ownershipOrControlStatement['interestedParty_unspecified_reason'].isna())}
    out = out.append(pd.DataFrame(d, index=['Missing']))
    out = sortCounts(out)
    return (out)
