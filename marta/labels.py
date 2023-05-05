def modify_labels(labels, organs):
    """
    Change labels so that the targetted sections are numbered from 1.0.
    """
    
    all_org = organs['all']
    main = organs['main']
    aux  = organs['aux']
    dict = organs['dict']
    
    # Modify the main labels to match the output of the main decoder
    main_labels = labels.clone()
    aux_labels = labels.clone()

    count_main = 1.0
    count_aux  = 1.0
    
    for organ in all_org:
        # Modify the labels to match the output of the decoder
        if organ in main:
            main_labels[main_labels == dict[organ]] = count_main
            count_main += 1.0
        else:
            main_labels[main_labels == dict[organ]] = 0.0
            
        if organ in aux:
            aux_labels[aux_labels == dict[organ]] = count_aux
            count_aux += 1.0
        else:
            aux_labels[aux_labels == dict[organ]] = 0.0
    
    return main_labels, aux_labels
