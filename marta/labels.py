def modify_labels(labels, organs_dict):
    # Modify the main labels to match the output of the main decoder
    main_labels = labels.clone()
    main_labels[(main_labels != organs_dict['Transition zone']) & (main_labels != organs_dict['Central gland'])] = 0.0
    main_labels[main_labels == organs_dict['Transition zone']] = 1.0
    main_labels[main_labels == organs_dict['Central gland']] = 2.0

    # Modify the auxilliary labels to match the output of the auxilliary decoder
    aux_labels = labels.clone()
    aux_labels[(aux_labels != organs_dict['Bladder']) & (aux_labels != organs_dict['Rectum']) & (aux_labels != organs_dict['Seminal vesicle'])] = 0.0
    aux_labels[aux_labels == organs_dict['Bladder']] = 1.0
    aux_labels[aux_labels == organs_dict['Rectum']] = 2.0
    aux_labels[aux_labels == organs_dict['Seminal vesicle']] = 3.0
    
    return main_labels, aux_labels
