def write_members_to_file(members, filename='members.txt'):
    """Write congressional member data to a text file."""
    with open(filename, 'w') as f:
        for row in members:
            f.write(','.join([str(value) for value in row]) + '\n')
