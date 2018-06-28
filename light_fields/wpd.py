def split_spectrums(f):
    """Separates a file with multiples spectrums, as exported from WebPlotDigitizer into files with single spectrums.

    Parameters:
    -----------
    f: str
        The path to the spectrum file.

    """
    name = os.path.splitext(f)[0]
    df = pd.read_csv(f, header=[0, 1])

    # Clean up
    prev, remove, ccts = None, list(), list()
    for column in df:
        if 'Unnamed' in column[0]:
            remove.append(column)
            df.loc[:, (prev, column[1])] = df[column[0]][column[1]]
        else:
            prev = column[0]
            ccts.append(column[0])
    df.drop(remove, axis=1, inplace=True)

    # Make a distribution for each CCT
    for cct in ccts:
        data = df.loc[:, cct].dropna()

        # Add missing extrema?
        if np.min(data['X']) > 400:
            data.loc[-1] = [400, 0]
        if np.max(data['X']) < 800:
            data.loc[len(data)] = [800, 0]

        data.sort_index().to_csv(name + " " + cct + '.csv', index=False, header=False)

    os.remove(f)
